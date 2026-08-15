import tempfile
import unittest
from pathlib import Path

from src.loader import canonical_name, extract_unit, load_log


class LoaderTests(unittest.TestCase):
    def test_dynamic_headers_are_unioned(self):
        text = (
            "Time (msec),Engine Speed (rpm),Mass Airflow (g/s)\n"
            "0,1000,10.0\n"
            "50,1100,11.0\n"
            "Time (msec),Engine Speed (rpm),Mass Airflow (g/s),IAM (1-byte)** (multiplier)\n"
            "100,1200,12.0,0.125\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dynamic.csv"
            path.write_text(text, encoding="utf-8")
            loaded = load_log(path)

        self.assertEqual(loaded.metadata.header_segments, 2)
        self.assertEqual(loaded.metadata.row_count, 3)
        self.assertIn("iam", loaded.data)
        self.assertEqual(loaded.data["iam"].notna().sum(), 1)
        self.assertAlmostEqual(loaded.metadata.median_period_s, 0.05)

    def test_known_name_and_unit(self):
        column = "Feedback Knock Correction (1-byte)** (degrees)"
        self.assertEqual(canonical_name(column), "fbkc")
        self.assertEqual(extract_unit(column), "degrees")


if __name__ == "__main__":
    unittest.main()
