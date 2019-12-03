import cv2


def find_largest_contour(contours):
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour
    return largest_contour


def blur_and_binarize(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    binarized_image = cv2.adaptiveThreshold(blurred_image, 255, 1, 1, 11, 2)
    return binarized_image


def crop_contour(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image