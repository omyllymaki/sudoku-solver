import logging
import os
import pandas as pd

import cv2

from digit_prediction import load_model
from sudoku_grabber import SudokuGrabber

logger = logging.getLogger(__name__)

IMAGE_PATH = os.path.join("data", "sudoku_images", "sudoku1.jpg")
OUTPUT_PATH = "sudoku_table.csv"
MODEL_PATH = os.path.join("models", "mnist_cnn.pt")
SIZE_TOLERANCE = 0.5
MIN_N_PIXELS = 5
PROBABILITY_THRESHOLD = 0.99
PLOT_RESULTS = True
DILATE_SIZES = range(8)

model = load_model(MODEL_PATH)


def main():
    image = cv2.imread(IMAGE_PATH)
    model = load_model(MODEL_PATH)
    grabber = SudokuGrabber(model, DILATE_SIZES, PROBABILITY_THRESHOLD, MIN_N_PIXELS, SIZE_TOLERANCE)
    sudoku_table = grabber.convert(image)

    df_sudoku = pd.DataFrame(sudoku_table)
    df_sudoku.to_csv(OUTPUT_PATH, index=False, header=False)

    print(df_sudoku)


if __name__ == '__main__':
    main()
