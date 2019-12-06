# TODO:
# check that there are no duplicate elements in rows, columns or blocks

import logging

import cv2
import numpy as np
import pandas as pd

from digit_prediction import load_model, predict_digit
from image_processing import find_largest_contour, crop_contour, binarize, dilate, erode, get_contour_coordinates, \
    show_images, binarize_adaptive

logger = logging.getLogger(__name__)

IMAGE_PATH = "data/sudoku1.jpg"
OUTPUT_PATH = "sudoku_table.csv"
MODEL_PATH = "mnist_cnn.pt"
SIZE_TOLERANCE = 0.5
MIN_N_PIXELS = 5
PROBABILITY_THRESHOLD = 0.99
PLOT_RESULTS = True
DILATE_SIZES = range(8)

model = load_model(MODEL_PATH)


def get_sudoku_image_contour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours)
    return largest_contour


def process_cell_image_for_analysis(cell_image, n_edge_pixels=3):
    cell = cv2.resize(cell_image, (28 + (2 * n_edge_pixels), 28 + (2 * n_edge_pixels)), interpolation=cv2.INTER_LINEAR)
    cell = cell[n_edge_pixels:-n_edge_pixels, n_edge_pixels:-n_edge_pixels]
    return cell


def analyze_cell_image(cell_image):
    n_white_pixels = sum(sum(cell_image > 250))
    if n_white_pixels > MIN_N_PIXELS:
        predicted_digit, probability = predict_digit(cell_image, model)
    else:
        predicted_digit = None
        probability = 1
    probability = round(probability, 2)
    return predicted_digit, probability


def get_cell_contours(image):
    height, width = image.shape
    expected_cell_area = int(height / 9) * int(width / 9)
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cell_contours = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        is_small_enough = (area < (1 + SIZE_TOLERANCE) * expected_cell_area)
        is_large_enough = (area > (1 - SIZE_TOLERANCE) * expected_cell_area)
        is_cell = is_large_enough and is_small_enough
        if is_cell:
            cell_contours.append(contour)
    return cell_contours


def create_digit_table(cells):
    cells = sorted(cells, key=lambda k: k['y'])
    digit_table, probability_table = [], []
    for k in range(9):
        row = cells[9 * (k):9 * (k + 1)]
        row = sorted(row, key=lambda k: k['x'])
        digits = [item["digit"] for item in row]
        probabilities = [item["probability"] for item in row]
        digit_table.append(digits)
        probability_table.append(probabilities)

    digit_table = np.array(digit_table)
    probability_table = np.array(probability_table)

    return digit_table, probability_table


def replace_uncertain_digits_with_question_mark(digit_table, probability_table):
    uncertain_elements = probability_table < PROBABILITY_THRESHOLD
    n_uncertain_digits = sum(sum(uncertain_elements))
    if n_uncertain_digits > 0:
        logger.warning(f"Extracted Sudoku table contains {n_uncertain_digits} uncertain digits")
    final_table = digit_table
    final_table[uncertain_elements] = "?"
    return final_table


def get_sudoku_cell_contours(input_image, dilate_sizes):
    for dilate_size in dilate_sizes:
        image = input_image.copy()
        image = dilate(image, (dilate_size, dilate_size))
        cell_contours = get_cell_contours(image)
        if len(cell_contours) == 81:
            return cell_contours
    show_images()
    raise Exception("Cannot find correct number of cells (81) from image")

def analyze_cell_contours(cell_contours, image):
    cell_data = []
    for cell_contour in cell_contours:
        x, y, _, _ = get_contour_coordinates(cell_contour)
        cell_image = crop_contour(image, cell_contour)
        cell_image = process_cell_image_for_analysis(cell_image)
        digit, probability = analyze_cell_image(cell_image)
        cell_data.append(dict(x=x, y=y, digit=digit, probability=probability))
        cell_image = cv2.resize(cell_image, (218, 218), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(f"{digit} {probability}", cell_image)
    return cell_data


def main(image_path):
    image = cv2.imread(image_path)
    cv2.imshow("Original image", image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image
    binary_image = ~binarize_adaptive(gray_image, 11, 15)
    cv2.imshow("Binary image", binary_image)

    binary_image_dilated = dilate(binary_image, (1,1))
    cv2.imshow("Binary image dilated", binary_image_dilated)

    sudoku_contour = get_sudoku_image_contour(binary_image_dilated)
    sudoku_image = crop_contour(binary_image, sudoku_contour)
    cv2.imshow("Sudoku", sudoku_image)

    cell_contours = get_sudoku_cell_contours(sudoku_image, DILATE_SIZES)

    cell_data = analyze_cell_contours(cell_contours, sudoku_image)
    digit_table, probability_table = create_digit_table(cell_data)
    sudoku_table = replace_uncertain_digits_with_question_mark(digit_table, probability_table)
    df_sudoku = pd.DataFrame(sudoku_table)
    df_sudoku.to_csv(OUTPUT_PATH, index=False, header=False)

    print(df_sudoku)
    show_images()


if __name__ == '__main__':
    main(IMAGE_PATH)
