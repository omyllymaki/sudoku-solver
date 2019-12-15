import cv2
import os
import unittest

from digit_classifier.digit_classifier import DigitClassifier
from sudoku_grabber.sudoku_grabber import SudokuGrabber

SUDOKU_IMAGES_DIR = os.path.join("data", "sudoku_images")
MODEL_PATH = os.path.join("data", "models", "mnist_cnn.pt")
SIZE_TOLERANCE = 0.5
MIN_N_PIXELS = 15
INTENSITY_THRESHOLD = 50
PROBABILITY_THRESHOLD = 0.5
DILATE_SIZES = range(8)


class SudokuGrabberTest(unittest.TestCase):
    solver = None

    def setUp(self):
        model = DigitClassifier(MODEL_PATH, min_n_white_pixels=MIN_N_PIXELS, threshold=INTENSITY_THRESHOLD)
        self.grabber = SudokuGrabber(model, DILATE_SIZES, PROBABILITY_THRESHOLD, SIZE_TOLERANCE, False, False)
        self.image_paths = self.get_image_paths()
        print(self.image_paths)

    def test_that_grabber_finds_correct_number_of_cells_from_images(self):
        for path in self.image_paths:
            image = cv2.imread(path)
            sudoku = self.grabber.convert(image)
            actual = sudoku.shape
            expected = (9, 9)
            self.assertEquals(actual, expected)

    @staticmethod
    def get_image_paths():
        file_paths = [os.path.join(SUDOKU_IMAGES_DIR, f)
                      for f in os.listdir(SUDOKU_IMAGES_DIR)
                      if f.endswith(".JPG")]
        return file_paths
