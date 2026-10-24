import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.analysis import _knock_timing_summary


class KnockTimingSummaryTests(unittest.TestCase):
    def test_counts_only_adjacent_loaded_correction_steps(self):
        data = pd.DataFrame(
            {
                "time_s": [0.00, 0.05, 0.10, 0.15, 0.20],
                "rpm": [2000, 2050, 2100, 2150, 2200],
                "throttle": [50, 50, 50, 50, 50],
                "knock_sum": [0, 0, 1, 1, 1],
                "flkc": [0, 0, -3, -3, 0],
                "knock_correction_advance": [0, 0, -3, -3, 0],
                "ignition_timing": [20, 20, 12, 12, 20],
            }
        )
        log = SimpleNamespace(data=data, metadata=SimpleNamespace(path=Path("sample.csv")))

        row = _knock_timing_summary([log]).iloc[0]

        self.assertEqual(row["knock_sum_increases"], 1)
        self.assertEqual(row["flkc_large_steps"], 2)
        self.assertEqual(row["kca_large_steps"], 2)
        self.assertEqual(row["ignition_large_steps"], 2)


if __name__ == "__main__":
    unittest.main()
