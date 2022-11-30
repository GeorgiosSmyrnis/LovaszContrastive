from typing import Optional, List, Tuple

import math
import numpy as np
from numba import njit

@njit
def get_colorjitter_params(
        brightness: Optional[np.ndarray],
        contrast: Optional[np.ndarray],
        saturation: Optional[np.ndarray],
        hue: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = np.random.permutation(4)

        b = None if brightness is None else float(np.random.uniform(brightness[0], brightness[1]))
        c = None if contrast is None else float(np.random.uniform(contrast[0], contrast[1]))
        s = None if saturation is None else float(np.random.uniform(saturation[0], saturation[1]))
        h = None if hue is None else float(np.random.uniform(hue[0], hue[1]))

        return fn_idx, b, c, s, h


@njit
def rgb_to_grayscale(img):
    result = np.zeros_like(img)
    result[:,:,0] = 0.2989 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    result[:,:,2] = result[:,:,1] = result[:,:,0]
    return result

@njit
def rgb_to_hsv(img):
    red = img[:,:,0]
    green = img[:,:,1]
    blue = img[:,:,2]

    xmax = np.maximum(red, green)
    xmax = np.maximum(xmax, blue)

    xmin = np.minimum(red, green)
    xmin = np.minimum(xmin, blue)

    value = xmax
    chroma = value - xmin

    hue = np.zeros_like(value)

    # Choose cases for hue.
    hue = np.where((red >= green) * (red >= blue) * (chroma > 0), (((0.0 + (green - blue) / chroma) / 6.0) % 1.0), hue)

    hue = np.where((green >= red) * (green >= blue) * (chroma > 0), (((2.0 + (blue - red) / chroma) / 6.0) % 1.0), hue)

    hue = np.where((blue >= red) * (blue >= green) * (chroma > 0), (((4.0 + (red - green) / chroma) / 6.0) % 1.0), hue)

    saturation = np.where(value > 0, chroma / value, np.zeros_like(value))

    out = np.zeros_like(img)
    out[:,:,0] = hue
    out[:,:,1] = saturation
    out[:,:,2] = value

    return out

@njit
def hsv_to_rgb(img):
    hue = img[:,:,0]
    saturation = img[:,:,1]
    value = img[:,:,2]

    chroma = saturation * value
    hue = hue * 6.0
    modulo = hue % 2.0
    x = chroma * (1 - np.abs(modulo - 1))

    red = np.zeros_like(hue)
    green = np.zeros_like(hue)
    blue = np.zeros_like(hue)

    mask = (hue < 1)
    red = red + mask * chroma
    green = green + mask * x

    mask = ((hue >= 1) * (hue < 2))
    red = red + mask * x
    green =  green + mask * chroma

    mask = ((hue >= 2) * (hue < 3))
    green = green + mask * chroma
    blue = blue + mask * x

    mask = ((hue >= 3) * (hue < 4))
    green = green + mask * x
    blue = blue + mask * chroma

    mask = ((hue >= 4) * (hue < 5))
    red = red + mask * x
    blue = blue + mask * chroma

    mask = (hue >= 5)
    red = red + mask * chroma
    blue = blue + mask * x

    out = np.zeros_like(img)
    out[:,:,0] = red
    out[:,:,1] = green
    out[:,:,2] = blue

    return out

@njit
def blend(img1, img2, factor):
    return np.clip((factor * img1 + (1.0 - factor) * img2), 0.0, 1.0)

@njit
def change_brightness(img, factor):
    return blend(img, np.zeros_like(img), factor)

@njit 
def change_contrast(img, factor):
    gray_image = rgb_to_grayscale(img)
    mean = np.mean(gray_image)
    mean_image = np.full_like(img, mean)
    result = blend(img, mean_image, factor)
    return result

@njit
def change_saturation(img, factor):
    gray_image = rgb_to_grayscale(img)
    return blend(img, gray_image, factor)

@njit
def change_hue(img, factor):
    img = rgb_to_hsv(img)
    img[:, :, 0] = (img[:, :, 0] + factor) % 1.0
    img = hsv_to_rgb(img)
    return img