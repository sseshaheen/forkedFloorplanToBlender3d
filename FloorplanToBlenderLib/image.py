import os
import cv2
import numpy as np
import logging
from PIL import Image
from . import calculate
from . import const
import matplotlib.pyplot as plt
from .globalConf import DEBUG_MODE, DEBUG_STORAGE_PATH, LOGGING_VERBOSE

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

"""
Image
This file contains code for image processing, used when creating blender project.
Contains functions for tweeking and filter images for better results.

FloorplanToBlender3d
Copyright (C) 2022 Daniel Westberg
"""

def save_image(title, img):
    """
    Save image to the debug directory.

    @param title: Title of the image, used as the filename.
    @param img: Image to be saved.
    """
    if DEBUG_MODE:
        filepath  = os.path.join(DEBUG_STORAGE_PATH, f"{title}.png")
        cv2.imwrite(filepath, img)
        if LOGGING_VERBOSE:
            logging.debug(f'Saved debug image: {filepath}')

def pil_rescale_image(image, factor):
    """
    Rescale a PIL image by a given factor.

    @param image: PIL Image to be rescaled.
    @param factor: Scaling factor.
    @return: Rescaled PIL Image.
    """
    width, height = image.size
    rescaled_image = image.resize((int(width * factor), int(height * factor)), resample=Image.BOX)
    if LOGGING_VERBOSE:
        logging.debug(f'Rescaled PIL image by a factor of {factor}')
    return rescaled_image

def cv2_rescale_image(image, factor):
    """
    Rescale an OpenCV image by a given factor.

    @param image: OpenCV Image to be rescaled.
    @param factor: Scaling factor.
    @return: Rescaled OpenCV Image.
    """
    rescaled_image = cv2.resize(image, None, fx=factor, fy=factor)
    if LOGGING_VERBOSE:
        logging.debug(f'Rescaled OpenCV image by a factor of {factor}')
    return rescaled_image

def pil_to_cv2(image):
    """
    Convert a PIL image to an OpenCV image.

    @param image: PIL Image to be converted.
    @return: Converted OpenCV Image.
    """
    cv2_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if LOGGING_VERBOSE:
        logging.debug('Converted PIL image to OpenCV image')
    return cv2_image

def calculate_scale_factor(preferred: float, value: float):
    """
    Calculate the scale factor based on preferred and actual values.

    @param preferred: Preferred value.
    @param value: Actual value.
    @return: Scale factor.
    """
    scale_factor = preferred / value
    if LOGGING_VERBOSE:
        logging.debug(f'Calculated scale factor: {scale_factor}')
    return scale_factor

def denoising(img, caller=None):
    """
    Apply denoising to an image.

    @param img: OpenCV Image to be denoised.
    @return: Denoised OpenCV Image.
    """
    denoised_img = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        const.IMAGE_H,
        const.IMAGE_HCOLOR,
        const.IMAGE_TEMPLATE_SIZE,
        const.IMAGE_SEARCH_SIZE,
    )
    if LOGGING_VERBOSE:
        logging.debug('Applied denoising to image')
    save_image(f'{caller}-denoised_image', denoised_img)
    return denoised_img

def remove_noise(img, noise_removal_threshold, caller=None):
    """
    Remove noise from an image and return the mask.

    @param img: Image to remove noise from.
    @param noise_removal_threshold: Threshold for noise removal.
    @return: Mask of the image with noise removed.
    """
    img[img < 128] = 0
    img[img > 128] = 255
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > noise_removal_threshold:
            cv2.fillPoly(mask, [contour], 255)
    if LOGGING_VERBOSE:
        logging.debug('Removed noise from image')
    save_image(f'{caller}-noise_removed_image', mask)
    return mask

def mark_outside_black(img, mask, caller=None):
    """
    Mark the outside of the image as black.

    @param img: Input image.
    @param mask: Mask to use.
    @return: Image with the outside marked black, updated mask.
    """
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, [biggest_contour], 255)
    img[mask == 0] = 0
    if LOGGING_VERBOSE:
        logging.debug('Marked outside of the image as black')
    save_image(f'{caller}-Image_with_Outside_Black', img)
    return img, mask

def detect_wall_rescale(reference_size, image):
    """
    Detect how much an image needs to be rescaled based on wall size.

    @param reference_size: Reference wall size to scale to.
    @param image: Input image.
    @return: Scale factor if walls are detected, otherwise None.
    """
    image_wall_size = calculate.wall_width_average(image, image_type='input_img')
    if LOGGING_VERBOSE:
        logging.debug(f'Reference size (based on the reference calibration image) is: {float(reference_size)}')
    if image_wall_size is None:
        logging.warning('Calculation of average wall width (used for checking if rescaling is needed) gave None')
        return None
    scale_factor = calculate_scale_factor(float(reference_size), image_wall_size)
    if LOGGING_VERBOSE:
        logging.debug(f'Calculation of average wall width gave: {image_wall_size}')
        logging.debug(f'Detected wall rescale factor: {scale_factor}')
    return scale_factor
