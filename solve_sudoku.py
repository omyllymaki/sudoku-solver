import os
import time

import numpy as np
import pandas as pd
import logging

from sudoku_solver.sudoku_solver import SudokuSolver

logging.basicConfig(level=logging.DEBUG)


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


PATH = os.path.join("data", "sudoku_tables", "sudoku_hardest.csv")
df_table = pd.read_csv(PATH, header=None, index_col=False)
table = df_table.to_numpy()
table = np.nan_to_num(table)
table = table.astype(int)
validate_table(table)

print(table)
solver = SudokuSolver()
time_start = time.time()
table = solver.solve(table)
time_end = time.time()
duration = time_end - time_start
print("-------------------")
print(table)
print(f"Solution took {duration:.3f} seconds")
