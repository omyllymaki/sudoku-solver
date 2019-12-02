import cv2
import numpy as np
import torch
from torchvision import transforms
import logging

from train_digit_classifier import Net

logger = logging.getLogger(__name__)

IMAGE_PATH = "data/sudoku1.jpg"
SIZE_TOLERANCE = 0.5
MIN_N_PIXELS = 10
PROBABILITY_THRESHOLD = 0.9

TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

model = Net()
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()


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


def extract_sudoku_table(image_path, plot_results = True):
    image = cv2.imread(image_path)
    if plot_results:
        cv2.imshow("Original image", image)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binarized_image = blur_and_binarize(gray_image)

    contours, _ = cv2.findContours(binarized_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours)

    sudoku_table_binarized = crop_contour(binarized_image, largest_contour)
    if plot_results:
        cv2.imshow("Sudoku table", sudoku_table_binarized)

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
            cell = cv2.resize(cell, (28, 28), interpolation=cv2.INTER_AREA)
            cell_center = cell[5:-5, 5:-5]
            n_white_pixels = sum(sum(cell_center > 250))

            if n_white_pixels > MIN_N_PIXELS:
                cell_tensor = TRANSFORMS(cell)
                cell_tensor = cell_tensor.unsqueeze(0)
                output = model(cell_tensor)
                predicted_digit = output.argmax(dim=1)[0].item()
                probability = round(torch.exp(output.max()).item(), 2)
            else:
                predicted_digit = None
                probability = 1

            cell_counter += 1
            cells.append(dict(x=x, y=y, digit=predicted_digit, probability=probability))

    if cell_counter != 81:
        logger.error(f"Sudoku game cannot be extracted from image. Number of cells found: {cell_counter}")

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

    uncertain_elements = probability_table < PROBABILITY_THRESHOLD
    if uncertain_elements.any():
        logger.warning("Extracted Sudoku table contains some uncertain elements that need to be filled manually.")
    final_table = digit_table
    final_table[uncertain_elements] = "?"
    print(final_table)

    if plot_results:
        cv2.imshow("Sudoku table new", sudoku_table_binarized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    extract_sudoku_table(IMAGE_PATH)
