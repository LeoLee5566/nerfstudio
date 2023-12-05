""" Helper functions for computing blurriness of outputs """

import numpy as np
import cv2
import torch

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

def get_svd_map(image:torch.Tensor, win_size:int=5, sv_num:int=2) -> torch.Tensor:
    device = image.device
    image = (image.cpu().numpy() * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pad_width = win_size // 2
    padded_image = np.pad(image, pad_width, mode='edge')
    result = np.zeros_like(image, dtype=float)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i+win_size, j:j+win_size]

            U, S, V = np.linalg.svd(window)

            ratio = np.sum(S[:sv_num]) / np.sum(S) if np.sum(S) != 0 else 1
            result[i, j] = ratio

    blur_map = (result-result.min())/(result.max()-result.min())
    return torch.from_numpy(blur_map)[:,:,None].to(device)
