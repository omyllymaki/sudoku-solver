import logging

import cv2
import numpy as np
import pandas as pd

from digit_prediction import load_model, predict_digit
from image_processing import find_largest_contour, process_image, crop_contour

logger = logging.getLogger(__name__)

IMAGE_PATH = "data/sudoku5.jpg"
OUTPUT_PATH = "sudoku_table.csv"
MODEL_PATH = "mnist_cnn.pt"
SIZE_TOLERANCE = 0.5
MIN_N_PIXELS = 5
PROBABILITY_THRESHOLD = 0.99
PLOT_RESULTS = True
DILATE_SIZES = range(8)

model = load_model(MODEL_PATH)


def extract_sudoku_image(image, dilate_size):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = process_image(gray_image, dilate_size=dilate_size)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours)

    binary_sudoku = crop_contour(binary_image, largest_contour)
    if PLOT_RESULTS:
        cv2.imshow("Sudoku table", binary_sudoku)

    return binary_sudoku


def analyze_cell(cell, n_edge_pixels=3):
    cell = cv2.resize(cell, (28 + (2 * n_edge_pixels), 28 + (2 * n_edge_pixels)), interpolation=cv2.INTER_AREA)
    cell = cell[n_edge_pixels:-n_edge_pixels, n_edge_pixels:-n_edge_pixels]
    n_white_pixels = sum(sum(cell > 250))
    if n_white_pixels > MIN_N_PIXELS:
        predicted_digit, probability = predict_digit(cell, model)
    else:
        predicted_digit = None
        probability = 1
    return predicted_digit, probability


def extract_cells(sudoku_table_binarized, erode_size=-1):
    sudoku_table_binarized = cv2.resize(sudoku_table_binarized, (500, 500), interpolation=cv2.INTER_AREA)
    expected_cell_area = int(500 / 9) * int(500 / 9)
    contours, _ = cv2.findContours(sudoku_table_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cells = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        is_cell = (area < (1 + SIZE_TOLERANCE) * expected_cell_area) and (
                area > (1 - SIZE_TOLERANCE) * expected_cell_area)
        if is_cell:
            cell = crop_contour(sudoku_table_binarized, contour)
            if erode_size > 0:
                kernel = np.ones((erode_size, erode_size), np.uint8)
                cell = cv2.erode(cell, kernel, iterations=1)
            predicted_digit, probability = analyze_cell(cell)
            cells.append(dict(x=x, y=y, digit=predicted_digit, probability=probability))

            if PLOT_RESULTS:
                cv2.imshow(f"cell {predicted_digit} {probability}", cell)
    return cells


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


def get_sudoku_cells(image, dilate_sizes):
    for dilate_size in dilate_sizes:
        sudoku_table_binarized = extract_sudoku_image(image, dilate_size)
        cells = extract_cells(sudoku_table_binarized)
        if len(cells) == 81:
            return cells
    raise Exception("Cannot find correct number of cells from image")


def main(image_path):
    image = cv2.imread(image_path)
    if PLOT_RESULTS:
        cv2.imshow("Original image", image)

    cells = get_sudoku_cells(image, DILATE_SIZES)
    digit_table, probability_table = create_digit_table(cells)
    final_table = replace_uncertain_digits_with_question_mark(digit_table, probability_table)
    df_final_table = pd.DataFrame(final_table)
    df_final_table.to_csv(OUTPUT_PATH, index=False, header=False)
    print(df_final_table)

    # TODO: check that there are no duplicate elements in rows, columns or blocks

    if PLOT_RESULTS:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(IMAGE_PATH)
