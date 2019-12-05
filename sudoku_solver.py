import pandas as pd
import numpy as np

PATH = "sudoku_easy.csv"


class SudokuSolver:

    def __init__(self):
        self.group_table = self.generate_group_table()
        self.possible_values = set(range(1,10))

    def solve(self, table):
        table = self.fill_certain(table)
        return table

    def fill_certain(self, table):
        n_nans_previous = sum(sum(np.isnan(table)))
        while True:

            for r in range(9):
                for c in range(9):
                    element = table[r, c]
                    if not np.isnan(element):
                        continue

                    row = table[r, :]
                    row_values = row[~np.isnan(row)]
                    column = table[:, c]
                    column_values = column[~np.isnan(column)]

                    element_group = self.group_table[r, c]
                    block = table[self.group_table == element_group]
                    block_values = block[~np.isnan(block)]

                    forbidden_values = set(row_values) | set(column_values) | set(block_values)
                    if len(forbidden_values) == 8:
                        only_possible_value = int(next(iter((self.possible_values - forbidden_values))))
                        table[r, c] = only_possible_value

            n_nans = sum(sum(np.isnan(table)))
            if (n_nans == 0) or (n_nans == n_nans_previous):
                return table

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



