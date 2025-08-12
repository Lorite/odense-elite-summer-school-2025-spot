import sys
import cv2 as cv
import numpy as np


def load_img(path):
    """
    Read image from path.

    Args:
        path: Path to image

    Returns:
        The image.

    Raises:
        AssertionError: If image could not be read.
    """
    img = cv.imread(path)
    assert img is not None
    return img


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
        return np.zeros(img.shape[:2], dtype=bool)
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


def draw_line(img, center, angle, color, thickness):
    """
    Draws a line across the entire image through center point at the specified angle.

    Args:
        img: The image to draw on.
        center: Center point the line should pass through.
        angle: Angle of the line.
        color: Color of the line.
        thickness: Thickness of the line.
    """
    theta = np.deg2rad(angle)

    img_h, img_w = img.shape[:2]
    length = int(np.sqrt(np.pow(img_w, 2) + np.pow(img_h, 2)))

    dx = np.cos(theta)
    dy = np.sin(theta)

    x1 = int(center[0] - dx * length)
    y1 = int(center[1] - dy * length)
    x2 = int(center[0] + dx * length)
    y2 = int(center[1] + dy * length)

    cv.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_rect(img, rect, color=(0, 0, 255), thickness=2):
    """
    Draws a rectangle with orientation lines and center point on the image.

    Args:
        img: The image to draw on.
        rect: The rectangle. May be None.
        color: Color of the line.
        thickness: Thickness of the line.
    """
    if rect is None:
        return

    box = cv.boxPoints(rect)
    box = np.intp(box)
    cv.drawContours(img, [box], 0, color, thickness)

    center = tuple(map(int, rect[0]))
    angle = rect[2]
    draw_line(img, center, angle, color, thickness)
    draw_line(img, center, angle+90, color, thickness)


def get_bgr_img_from_mask(mask):
    """
    Gets BGR image from bool mask.

    Args:
        mask: The mask. If no boolean mask is provided, the input is returned.

    Returns:
        The BGR image.
    """

    if mask.dtype == np.bool_ or mask.dtype == bool:
        img = (mask.astype(np.uint8)) * 255
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return mask


def show_imgs(imgs, rect=None):
    """
    Shows multiple images/masks side by side.

    Args:
        imgs: List of images/masks.
        rect: Rectangle to draw on all images. May be None
    """
    imgs_converted = []
    for img in imgs:
        img_converted = get_bgr_img_from_mask(img)
        draw_rect(img_converted, rect)
        imgs_converted.append(img_converted)
    combined = np.hstack(imgs_converted)
    cv.imshow("", combined)
    cv.waitKey(0)
    cv.destroyAllWindows()


############################################################################

def detect_object(img, img_blur=1, color_threshold=3.0, mask_opening_radius=1):
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
    return rect, (img_pp, color_mask, mask_pp, largest_component_mask)

############################################################################


if __name__ == '__main__':

    if len(sys.argv) != 5:
        print("Usage: python detect_object.py <img_path> <img_blur> <color_threshold> <mask_opening_radius>")
        exit(1)

    img_path = sys.argv[1]
    img_blur = int(sys.argv[2])
    color_threshold = float(sys.argv[3])
    mask_opening_radius = int(sys.argv[4])

    img = load_img(img_path)
    rect, (img_pp, color_mask, mask_pp, largest_component_mask) = detect_object(
        img, img_blur, color_threshold, mask_opening_radius)

    print(rect)

    show_imgs([img, img_pp, color_mask, mask_pp, largest_component_mask], rect)
