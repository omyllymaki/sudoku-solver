import time

import numpy as np
import pandas as pd

GROUP_TABLE = np.array([
    [0, 0, 0, 1, 1, 1, 2, 2, 2],
    [0, 0, 0, 1, 1, 1, 2, 2, 2],
    [0, 0, 0, 1, 1, 1, 2, 2, 2],
    [3, 3, 3, 4, 4, 4, 5, 5, 5],
    [3, 3, 3, 4, 4, 4, 5, 5, 5],
    [3, 3, 3, 4, 4, 4, 5, 5, 5],
    [6, 6, 6, 7, 7, 7, 8, 8, 8],
    [6, 6, 6, 7, 7, 7, 8, 8, 8],
    [6, 6, 6, 7, 7, 7, 8, 8, 8]
])


def validate_table(table):
    w, h = table.shape
    if w != 9:
        raise Exception(f"Invalid table width: {w}")
    if h != 9:
        raise Exception(f"Invalid table height: {h}")

    for i in range(9):
        row_values = table[i, :]
        row_values = row_values[row_values != 0]
        if len(row_values) != len(set(row_values)):
            raise Exception(f"Row {i} contains duplicate elements")

    for i in range(9):
        column_values = table[:, i]
        column_values = column_values[column_values != 0]
        if len(column_values) != len(set(column_values)):
            raise Exception(f"Column {i} contains duplicate elements")

    unique_values = np.unique(table)
    for value in unique_values:
        if value not in list(range(0, 10)):
            raise Exception(f"Table contains invalid value: {value}")


def print_board(table):
    print(np.array(table))


def find_empty_cells(table):
    return np.argwhere(table == 0)


def get_possible_values(table, r, c):
    row_values = table[r, :]
    column_values = table[:, c]

    element_group = GROUP_TABLE[r, c]
    block_values = table[GROUP_TABLE == element_group]

    forbidden_values = set(row_values) | set(column_values) | set(block_values)
    possible_values = set(range(1, 10)) - forbidden_values

    return list(possible_values)


def create_book_keeper(table):
    empty_cells = find_empty_cells(table)
    book_keeper = []
    for empty_cell in empty_cells:
        item = (empty_cell[0], empty_cell[1], 0)
        book_keeper.append(item)
    return book_keeper


def solve(table):
    book_keeper = create_book_keeper(table)

    counter, index = 0, 0
    while True:

        if index == len(book_keeper):
            print(f"Solution took {counter} rounds")
            return True

        if index < 0:
            print(f"Solution cannot be found")
            return False

        counter += 1

        r, c, current_value = book_keeper[index]
        possible_values = get_possible_values(table, r, c)
        possible_values = [v for v in possible_values if v > current_value]
        if not possible_values:
            book_keeper[index] = (r, c, 0)
            table[r, c] = 0
            index -= 1
            continue

        new_value = min(possible_values)
        book_keeper[index] = (r, c, new_value)
        table[r, c] = new_value
        index += 1


PATH = "sudoku_hardest.csv"
df_table = pd.read_csv(PATH, header=None, index_col=False)
table = df_table.to_numpy()
table = np.nan_to_num(table)
table = table.astype(int)
validate_table(table)

print(table)
time_start = time.time()
solve(table)
time_end = time.time()
duration = time_end - time_start
print("-------------------")
print(table)
print(f"Solution took {duration:.3f} seconds")
