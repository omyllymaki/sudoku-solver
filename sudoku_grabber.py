import logging

import cv2
import numpy as np
import pandas as pd

from digit_prediction import load_model, predict_digit
from image_processing import find_largest_contour, blur_and_binarize, crop_contour

logger = logging.getLogger(__name__)

IMAGE_PATH = "data/sudoku3.jpg"
OUTPUT_PATH = "sudoku_table.csv"
MODEL_PATH = "mnist_cnn.pt"
SIZE_TOLERANCE = 0.5
MIN_N_PIXELS = 10
PROBABILITY_THRESHOLD = 0.99
PLOT_RESULTS = True

model = load_model(MODEL_PATH)


def extract_sudoku_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binarized_image = blur_and_binarize(gray_image)

    contours, _ = cv2.findContours(binarized_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours)

    sudoku_table_binarized = crop_contour(binarized_image, largest_contour)
    if PLOT_RESULTS:
        cv2.imshow("Sudoku table", sudoku_table_binarized)

    return sudoku_table_binarized


def extract_cells(sudoku_table_binarized):
    sudoku_table_binarized = cv2.resize(sudoku_table_binarized, (500, 500), interpolation=cv2.INTER_AREA)
    expected_cell_area = int(500 / 9) * int(500 / 9)

    contours, _ = cv2.findContours(sudoku_table_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cell_counter = 0
    cells = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        is_cell = (area < (1 + SIZE_TOLERANCE) * expected_cell_area) and (
                area > (1 - SIZE_TOLERANCE) * expected_cell_area)
        if is_cell:
            cell = crop_contour(sudoku_table_binarized, contour)
            cell = cv2.resize(cell, (34, 34), interpolation=cv2.INTER_AREA)
            cell = cell[3:-3, 3:-3]
            n_white_pixels = sum(sum(cell > 250))

            if n_white_pixels > MIN_N_PIXELS:
                predicted_digit, probability = predict_digit(cell, model)
            else:
                predicted_digit = None
                probability = 1

            cell_counter += 1
            cells.append(dict(x=x, y=y, digit=predicted_digit, probability=probability))

    if cell_counter != 81:
        raise Exception(f"Sudoku game cannot be extracted from image. Number of cells found: {cell_counter}")

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


def main(image_path):
    image = cv2.imread(image_path)
    if PLOT_RESULTS:
        cv2.imshow("Original image", image)

    sudoku_table_binarized = extract_sudoku_image(image)
    cells = extract_cells(sudoku_table_binarized)
    digit_table, probability_table = create_digit_table(cells)
    final_table = replace_uncertain_digits_with_question_mark(digit_table, probability_table)
    df_final_table = pd.DataFrame(final_table)
    df_final_table.to_csv(OUTPUT_PATH, index=False, header=False)
    print(df_final_table)

    if PLOT_RESULTS:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(IMAGE_PATH)
