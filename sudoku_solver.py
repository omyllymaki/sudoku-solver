import numpy as np
import pandas as pd

PATH = "sudoku_medium.csv"


class SudokuSolver:

    def __init__(self):
        self.group_table = self.generate_group_table()
        self.possible_values = set(range(1, 10))

    def solve(self, table):
        self.table = table

        self.fill_certain()
        if self.is_sudoku_solved():
            return self.table

    def get_cell_with_lowest_number_of_possible_values(self):
        min_possible_values = 0
        r_out, c_out = None, None

        for r in range(9):
            for c in range(9):
                element = self.table[r, c]
                if not np.isnan(element):
                    continue
                possible_values = self.get_possible_values(r, c)
                if len(possible_values) < min_possible_values:
                    r_out, c_out = r, c
                    min_possible_values = len(possible_values)
        return r_out, c_out

    def fill_certain(self):
        n_nans_previous = sum(sum(np.isnan(self.table)))
        while True:

            for r in range(9):
                for c in range(9):
                    element = self.table[r, c]
                    if not np.isnan(element):
                        continue

                    possible_values = self.get_possible_values(r, c)
                    if len(possible_values) == 1:
                        table[r, c] = possible_values[0]

            n_nans = sum(sum(np.isnan(table)))
            if (n_nans == 0) or (n_nans == n_nans_previous):
                return table
            else:
                n_nans_previous = n_nans

    def get_possible_values(self, r, c):
        row = self.table[r, :]
        row_values = row[~np.isnan(row)]
        column = self.table[:, c]
        column_values = column[~np.isnan(column)]

        element_group = self.group_table[r, c]
        block = self.table[self.group_table == element_group]
        block_values = block[~np.isnan(block)]

        forbidden_values = set(row_values) | set(column_values) | set(block_values)
        possible_values = self.possible_values - forbidden_values

        return list(possible_values)

    def is_sudoku_solved(self):
        if sum(sum(np.isnan(self.table))) == 0:
            return True
        else:
            return False

    @staticmethod
    def generate_group_table():
        group = 0
        group_table = np.empty((0, 9))
        for i in range(3):
            row_table = np.empty((3, 0))
            for j in range(3):
                block = group * np.ones((3, 3))
                group += 1
                row_table = np.hstack((row_table, block))
            group_table = np.vstack((group_table, row_table))
        return group_table


def validate_table(table):
    w, h = table.shape
    if w != 9:
        raise Exception(f"Invalid table width: {w}")
    if h != 9:
        raise Exception(f"Invalid table height: {h}")

    for i in range(9):
        row = table[i, :]
        row_values = row[~np.isnan(row)]
        if len(row_values) != len(set(row_values)):
            raise Exception(f"Row {i} contains duplicate elements")

    for i in range(9):
        column = table[:, i]
        column_values = column[~np.isnan(column)]
        if len(column_values) != len(set(column_values)):
            raise Exception(f"Column {i} contains duplicate elements")

    unique_values = np.unique(table)
    unique_values = unique_values[~np.isnan(unique_values)]
    for value in unique_values:
        if value not in list(range(1, 10)):
            raise Exception(f"Table contains invalid value: {value}")


df_table = pd.read_csv(PATH, header=None, index_col=False)
table = df_table.to_numpy()
validate_table(table)
print(table)

solver = SudokuSolver()
table = solver.solve(table)
print(table)
