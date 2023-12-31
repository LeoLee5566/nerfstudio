""" Helper functions for computing blurriness of outputs """

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

def gaussian_kernel(size=3, sigma=1):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy, zz = np.meshgrid(ax, ax, ax)

    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))

    kernel = kernel / np.sum(kernel)

    return torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
  
def apply_kernel(tensor:torch.Tensor) -> torch.Tensor:
    device = tensor.device
    tensor_batched = tensor.unsqueeze(0)
    kernel = gaussian_kernel().to(device)

    result = F.conv3d(tensor_batched, kernel, padding=1)

    return result.squeeze(0)
  
def get_std_map(image:torch.Tensor,kernel_size:int=7) -> torch.Tensor:
  device = image.device
  image = (image.cpu().numpy() * 255).astype(np.uint8)
  if image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  laplacian = cv2.Laplacian(image, cv2.CV_64F)

  # Take the absolute value of the edge detection result to ensure non-negative values
  laplacian_abs = np.absolute(laplacian)
  # Compute local variance or standard deviation around each pixel
  local_variance_map = cv2.blur(laplacian_abs**2, (kernel_size, kernel_size)) - \
                      cv2.blur(laplacian_abs, (kernel_size, kernel_size))**2

  # The sqrt of the variance gives us the standard deviation, another measure of sharpness
  local_stddev_map = np.sqrt(local_variance_map)

  # Normalize the local standard deviation map for better visualization
  local_stddev_map_normalized = cv2.normalize(local_stddev_map, None, 0, 255, cv2.NORM_MINMAX)
  return torch.from_numpy(local_stddev_map_normalized)[:,:,None].to(device)


def get_svd_map(image:torch.Tensor, win_size:int=3, sv_num:int=1) -> torch.Tensor:
    device = image.device
    image = (image.cpu().numpy() * 255).astype(np.uint8)
    if image.shape[2] == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pad_width = win_size // 2
    padded_image = np.pad(image, pad_width, mode='edge')
    result = np.zeros_like(image, dtype=float)

    for i in trange(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i+win_size, j:j+win_size]

            U, S, V = np.linalg.svd(window)

            ratio = np.sum(S[:sv_num]) / np.sum(S) if np.sum(S) != 0 else 1
            result[i, j] = ratio

    blur_map = np.ones_like(image, dtype=float) - ((result-result.min())/(result.max()-result.min())) if result.max() != result.min() else np.zeros_like(image, dtype=float)
    return torch.from_numpy(blur_map)[:,:,None].to(device)
  
def get_svd_map_3D(sample:torch.Tensor,  win_size:int=3, sv_num:int=1):
    device = sample.device
    height, width, channel, _ = sample.shape

    singular_values = np.zeros((height, width, channel, win_size**2))

    for i in trange(height):
        for j in range(width):
            for c in range(channel):
                window = sample[i:i+win_size, j:j+win_size, c, 0]
                U, S, Vh = np.linalg.svd(window, full_matrices=False)
                singular_values[i, j, c, :len(S)] = S

    # Compute the top k SVD ratios
    result = np.zeros((height, width, channel))

    for i in range(height):
        for j in range(width):
            for c in range(channel):
                top_k_singular_values = singular_values[i, j, c, :sv_num]
                svd_ratio = np.sum(top_k_singular_values) / np.sum(singular_values[i, j, c])
                result[i, j, c] = svd_ratio
    result = result.unsqueeze(0)
    blur_map = np.ones_like(sample, dtype=float) - ((result-result.min())/(result.max()-result.min())) if result.max() != result.min() else np.zeros_like(sample, dtype=float)
    return torch.from_numpy(blur_map).to(device)