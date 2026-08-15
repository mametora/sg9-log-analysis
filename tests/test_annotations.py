import tempfile
import unittest
from pathlib import Path

from src.annotations import load_annotations


class AnnotationTests(unittest.TestCase):
    def test_omitted_file_returns_empty_annotations(self):
        annotations = load_annotations(None)
        self.assertIsNone(annotations.reported_symptom)
        self.assertEqual(annotations.inspection_windows, [])

    def test_json_annotations_are_loaded(self):
        text = """{
          "reported_symptom": {"log": "sample.csv", "near_end_s": 12},
          "inspection_windows": [
            {"log": "sample.csv", "start_s": 10, "end_s": 14, "label": "check"}
          ]
        }"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(text, encoding="utf-8")
            annotations = load_annotations(path)

        self.assertEqual(annotations.reported_symptom.log, "sample.csv")
        self.assertEqual(annotations.reported_symptom.near_end_s, 12)
        self.assertEqual(annotations.inspection_windows[0].label, "check")


if __name__ == "__main__":
    unittest.main()
