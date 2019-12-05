import numpy as np
import pandas as pd

PATH = "sudoku_medium.csv"


class SudokuSolver:

    def __init__(self):
        self.group_table = self.generate_group_table()
        self.possible_values = set(range(1, 10))

    def solve(self, table):
        self.table = table
        self.validate_table()
        self.fill_certain()
        self.validate_table()
        return self.table

    def validate_table(self):
        w, h = self.table.shape
        if w != 9:
            raise Exception(f"Invalid table width: {w}")
        if h != 9:
            raise Exception(f"Invalid table height: {h}")

        for i in range(9):
            row = self.table[i,:]
            row_values = row[~np.isnan(row)]
            if len(row_values) != len(set(row_values)):
                raise Exception(f"Row {i} contains duplicate elements")

        for i in range(9):
            column = self.table[:,i]
            column_values = column[~np.isnan(column)]
            if len(column_values) != len(set(column_values)):
                raise Exception(f"Column {i} contains duplicate elements")

        unique_values = np.unique(self.table)
        unique_values = unique_values[~np.isnan(unique_values)]
        for value in unique_values:
            if value not in self.possible_values:
                raise Exception(f"Table contains invalid value: {value}")


    def fill_certain(self):
        n_nans_previous = sum(sum(np.isnan(self.table)))
        while True:

            for r in range(9):
                for c in range(9):
                    element = self.table[r, c]
                    if not np.isnan(element):
                        continue

                    forbidden_values = self.get_forbidden_values(r,c)
                    if len(forbidden_values) == 8:
                        only_possible_value = int(next(iter((self.possible_values - forbidden_values))))
                        table[r, c] = only_possible_value

            n_nans = sum(sum(np.isnan(table)))
            if (n_nans == 0) or (n_nans == n_nans_previous):
                return table

    def get_forbidden_values(self, r, c):
        row = self.table[r, :]
        row_values = row[~np.isnan(row)]
        column = self.table[:, c]
        column_values = column[~np.isnan(column)]

        element_group = self.group_table[r, c]
        block = self.table[self.group_table == element_group]
        block_values = block[~np.isnan(block)]

        forbidden_values = set(row_values) | set(column_values) | set(block_values)

        return forbidden_values

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


df_table = pd.read_csv(PATH, header=None, index_col=False)
table = df_table.to_numpy()
print(table)

solver = SudokuSolver()
table = solver.solve(table)
print(table)
