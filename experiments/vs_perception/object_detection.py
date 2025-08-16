import cv2 as cv
import numpy as np


def preprocess_img(img, img_blur):
    """
    Get preprocessed image. Converts color to LAB space and applies blur.

    Args:
        img: The image to preprocess.
        img_blur: Radius of blur to apply to the image.
                  Must be int >= 1. Blur 1 has no effect.

    Returns:
        Preprocessed image.
    """
    img_pp = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    img_pp = cv.blur(img_pp, (img_blur, img_blur))
    return img_pp


def get_color_mask(img, color_threshold):
    """
    Get mask of pixels deviating beyond a threshold from the median color.

    Args:
        img: The image.
        color_threshold: Threshold for pixels to mask. Must be > 0.

    Returns:
        Boolean mask.
    """
    pixels = img.reshape((-1, 3))
    median_color = np.median(pixels, axis=0)
    diff = np.linalg.norm(pixels - median_color, axis=1)
    threshold_value = np.mean(diff) + color_threshold * np.std(diff)
    mask = (diff > threshold_value).reshape(img.shape[:2])
    return mask


def postprocess_mask(mask, mask_opening_radius):
    """
    Get postprocessed mask. Applies morphological opening.

    Args:
        mask: Boolean mask.
        mask_opening_radius: Radius of kernel used for morphological opening.
                             Must be int > 0.

    Returns:
        Postprocessed boolean mask.
    """
    mask_uint8 = (mask.astype(np.uint8)) * 255
    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (mask_opening_radius, mask_opening_radius))
    opened = cv.morphologyEx(mask_uint8, cv.MORPH_OPEN, kernel)
    return opened > 0


def get_mask_largest_component(mask):
    """
    Get mask only containing the largest connected component.

    Args:
        mask: Boolean mask.

    Returns:
        Boolean mask only containing the largest connected component. If input
        mask only contains False, output will also be mask containting only False.
    """
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=4)
    if num_labels == 1:
        return np.zeros(mask.shape[:2], dtype=bool)
    largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
    return (labels == largest_label).astype(bool)


def get_min_area_rect(mask):
    """
    Get min area rectangle containing all True pixels in boolean mask.

    Args:
        mask: Boolean mask.

    Returns:
        Min area rectangle in format ((center_x, center_y), (width, height), angle).
        None if mask only containing False pixels is provided.
    """
    if not mask.any():
        return None
    coords = np.column_stack(np.where(mask))
    coords = coords[:, ::-1]
    rect = cv.minAreaRect(coords)
    return rect


class DetectObjectResult():

    def __init__(self, img_pp, color_mask, mask_pp, largest_component_mask, rect):
        self.rect = rect
        self.img_pp = img_pp
        self.color_mask = color_mask
        self.mask_pp = mask_pp
        self.largest_component_mask = largest_component_mask


def detect_object(img, img_blur, color_threshold, mask_opening_radius):
    """
    Detects largest object with distinct color from background.

    Args:
        img: The image.
        img_blur: Radius of blur to apply in preprocessing. Must be int > 0.
        color_threshold: Threshold for pixels to mask. Must be > 0.
        mask_opening_radius: Radius of kernel used for morphological opening for 
                             mask postprocessing. Must be int > 0.

    Returns:
        rect: Min area around detected object rectangle in format 
              ((center_x, center_y), (width, height), angle). None
              if no object was detected.
        img_pp: Preprocessed image.
        color_mask: Mask from color deviation detection
        mask_pp: Postprocessed mask after morphological opening
        largest_component_mask: Mask only containing largest detected object.
    """
    img_pp = preprocess_img(img, img_blur)
    color_mask = get_color_mask(img_pp, color_threshold)
    mask_pp = postprocess_mask(color_mask, mask_opening_radius)
    largest_component_mask = get_mask_largest_component(mask_pp)
    rect = get_min_area_rect(largest_component_mask)

    return DetectObjectResult(img_pp, color_mask, mask_pp, largest_component_mask, rect)
