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


def process_image(image, gaussian_size=5, dilate_size=3, erode_size=0):
    image = cv2.GaussianBlur(image, (gaussian_size, gaussian_size), 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 1, 11, 2)
    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return image


def crop_contour(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image