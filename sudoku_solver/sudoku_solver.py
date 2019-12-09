import logging

import numpy as np

logger = logging.getLogger(__name__)


class SudokuSolver:
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

    def solve(self, table):
        empty_cells = self._find_empty_cells(table)

        counter, index = 0, 0
        while True:

            if index == len(empty_cells):
                logger.info(f"Solution found and took {counter} rounds")
                return table

            if index < 0:
                logger.info(f"Solution cannot be found")
                return None

            r, c = empty_cells[index]
            current_value = table[r, c]
            possible_values = self._get_possible_values(table, r, c)
            possible_values = [v for v in possible_values if v > current_value]
            if possible_values:
                table[r, c] = min(possible_values)
                index += 1
            else:
                table[r, c] = 0
                index -= 1

            counter += 1

    def _get_possible_values(self, table, r, c):
        row_values = table[r, :]
        column_values = table[:, c]

        element_group = self.GROUP_TABLE[r, c]
        block_values = table[self.GROUP_TABLE == element_group]

        forbidden_values = set(row_values) | set(column_values) | set(block_values)
        possible_values = set(range(1, 10)) - forbidden_values

        return list(possible_values)

    @staticmethod
    def _find_empty_cells(table):
        return np.argwhere(table == 0)
