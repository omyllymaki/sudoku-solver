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


def find_contour_corners(contour):
    # Bottom-right point has the largest (x + y) value
    # Top-left has point smallest (x + y) value
    # Bottom-left point has smallest (x - y) value
    # Top-right point has largest (x - y) value

    result = np.array([None, None, None, None])  # [top_left, top_right, bottom_right, bottom_left]
    x_plus_y_max, x_minus_y_max = -np.inf, -np.inf
    x_plus_y_min, x_minus_y_min = np.inf, np.inf
    for index, item in enumerate(contour):
        x = item[0][0]
        y = item[0][1]

        x_plus_y = x + y
        x_minus_y = x - y

        if x_plus_y < x_plus_y_min:
            result[0] = np.array([x,y])
            x_plus_y_min = x_plus_y

        if x_minus_y > x_minus_y_max:
            result[1] = np.array([x,y])
            x_minus_y_max = x_minus_y

        if x_plus_y > x_plus_y_max:
            result[2] = np.array([x,y])
            x_plus_y_max = x_plus_y

        if x_minus_y < x_minus_y_min:
            result[3] = np.array([x,y])
            x_minus_y_min = x_minus_y

    result = np.array([result[0], result[1], result[2], result[3]], dtype='float32')
    return result


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


