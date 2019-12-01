import cv2
import pytesseract

IMAGE_PATH = "data/sudoku1.jpg"
SIZE_TOLERANCE = 0.5


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


image = cv2.imread(IMAGE_PATH)
cv2.imshow("Original image", image)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binarized_image = blur_and_binarize(gray_image)

contours, _ = cv2.findContours(binarized_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = find_largest_contour(contours)

sudoku_table = crop_contour(binarized_image, largest_contour)
sudoku_table_binarized = crop_contour(binarized_image, largest_contour)
cv2.imshow("Sudoku table", sudoku_table_binarized)

sudoku_table_binarized = cv2.resize(sudoku_table_binarized, (500, 500), interpolation=cv2.INTER_AREA)
expected_cell_area = int(500 / 9) * int(500 / 9)

contours, _ = cv2.findContours(sudoku_table_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

n_cells = 0
cells = []
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h
    is_cell = (area < (1 + SIZE_TOLERANCE) * expected_cell_area) and (area > (1 - SIZE_TOLERANCE) * expected_cell_area)
    if is_cell:
        cell = crop_contour(sudoku_table_binarized, contour)
        config = ("--oem 2 --psm 10")
        detected_digit = pytesseract.image_to_string(cell, lang="eng", config=config)
        print(detected_digit)
        n_cells += 1
        cells.append(dict(image=cell, x=x, y=y))

if n_cells != 81:
    print("Sudoku game cannot be extracted from image")
    print(f"Number of cells found: {n_cells}")

cv2.imshow("Sudoku table new", sudoku_table_binarized)

cv2.waitKey(0)
cv2.destroyAllWindows()

