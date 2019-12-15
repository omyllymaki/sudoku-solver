import logging
import os
import pandas as pd

import cv2

from digit_classifier.digit_classifier import DigitClassifier
from sudoku_grabber.sudoku_grabber import SudokuGrabber

logger = logging.getLogger(__name__)

IMAGE_PATH = os.path.join("data", "sudoku_images", "sudoku4.jpg")
OUTPUT_PATH = "sudoku_table.csv"
MODEL_PATH = os.path.join("models", "mnist_cnn.pt")
SIZE_TOLERANCE = 0.5
MIN_N_PIXELS = 15
INTENSITY_THRESHOLD = 50
PROBABILITY_THRESHOLD = 0.5
PLOT_RESULTS = True
DILATE_SIZES = range(8)


def main():
    image = cv2.imread(IMAGE_PATH)
    model = DigitClassifier(MODEL_PATH, min_n_white_pixels=MIN_N_PIXELS, threshold=INTENSITY_THRESHOLD)
    grabber = SudokuGrabber(model, DILATE_SIZES, PROBABILITY_THRESHOLD, SIZE_TOLERANCE, True, True)
    sudoku_table = grabber.convert(image)

    df_sudoku = pd.DataFrame(sudoku_table)
    df_sudoku.to_csv(OUTPUT_PATH, index=False, header=False)

    print(df_sudoku)


if __name__ == '__main__':
    main()
