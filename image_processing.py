import cv2
import numpy as np


def find_largest_contour(contours):
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour
    return largest_contour


def crop_contour(image, contour):
    x, y, w, h = get_contour_coordinates(contour)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image


def get_contour_coordinates(contour):
    return cv2.boundingRect(contour)


def binarize(input_image, threshold=None):
    image = input_image.copy()
    if not threshold:
        threshold = np.mean(image)
    iw = image >= threshold
    ib = image < threshold
    image[iw] = 255
    image[ib] = 0
    return image


def dilate(input_image, dimensions):
    image = input_image.copy()
    kernel = np.ones(dimensions, np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def erode(input_image, dimensions):
    image = input_image.copy()
    kernel = np.ones(dimensions, np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def binarize_adaptive(image, block_size=11, offset=2):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, offset)


def show_images():
    cv2.waitKey(0)
    cv2.destroyAllWindows()
