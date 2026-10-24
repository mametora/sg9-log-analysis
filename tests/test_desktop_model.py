import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.desktop.model import (Annotation, Document, Panel, Workspace,
                               export_segment, time_range, write_json)


CSV = """Time (msec),Engine Speed (rpm),Manifold Relative Pressure (kPa)
1000,1000,-40
1100,2000,-20
1200,,0
1300,4000,20
Time (msec),Engine Speed (rpm),Manifold Relative Pressure (kPa),Mass Airflow (g/s)
1400,5000,40,80
3000,6000,60,100
3100,7000,80,120
100,8000,90,140
200,9000,95,150
"""


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "source.csv"
        self.path.write_text(CSV)
        self.doc = Document.open(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_range_statistics_use_original_samples_and_comparison_time(self):
        self.doc.anchor = .1
        stats = self.doc.statistics("rpm", (0, .31))
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["min"], 2000)
        self.assertEqual(stats["max"], 5000)
        self.assertAlmostEqual(stats["mean"], 11000/3)
        self.assertEqual(stats["median"], 4000)
        self.assertEqual(self.doc.statistics("rpm", (50, 60))["count"], 0)
        self.assertEqual(self.doc.statistics("rpm", None)["count"], 8)

    def test_plot_breaks_preserve_missing_values_gaps_and_resets(self):
        x, y = self.doc.plot_arrays("rpm")
        self.assertEqual(np.isnan(x).sum(), 2)
        self.assertEqual(np.isnan(y).sum(), 3)
        self.assertEqual(len(y), len(self.doc.elapsed)+2)
        self.assertEqual(self.doc.values("rpm")[-1], 9000)
        self.assertTrue(any("巻き戻り" in w for w in self.doc.warnings()))

    def test_cursor_does_not_interpolate_across_missing_values_or_gaps(self):
        sample = self.doc.sample("rpm", .1)
        self.assertAlmostEqual(sample[0], .1)
        self.assertEqual(sample[1], 2000.0)
        self.assertIsNone(self.doc.sample("rpm", .2))
        self.assertIsNone(self.doc.sample("rpm", 1.0))
        self.assertIsNone(self.doc.sample("rpm", 999))

    def test_session_roundtrip_preserves_alignment_and_annotations(self):
        self.doc.anchor, self.doc.enabled = .1, False
        workspace = Workspace(documents=[self.doc], bounds=(-.1, .3), columns=2,
                              linked=False, cursors=[0, .2], stat_all=True,
                              annotations=[Annotation(0, .2, "確認", "テスト")])
        workspace.panels[0].statistics[self.doc.id+"|rpm"] = ["mean", "median"]
        workspace.panels[0].bounds = (-.1, .2)
        destination = self.root / "analysis.json"
        write_json(destination, workspace.session(destination), workspace.documents)
        data = json.loads(destination.read_text())
        self.assertEqual(data["documents"][0]["path"], "source.csv")
        reopened = Document.open(self.root / data["documents"][0]["path"])
        entry = data["documents"][0]
        reopened.id, reopened.anchor, reopened.enabled = entry["id"], entry["anchor"], entry["enabled"]
        restored = Workspace.restore(data, [reopened])
        self.assertEqual(restored.bounds, (-.1, .3))
        self.assertEqual(restored.documents[0].anchor, .1)
        self.assertFalse(restored.documents[0].enabled)
        self.assertEqual(restored.annotations[0].note, "テスト")
        self.assertEqual(restored.panels[0].statistics[self.doc.id+"|rpm"], ["mean", "median"])
        self.assertEqual(restored.panels[0].bounds, (-.1, .2))
        self.assertTrue(restored.stat_all)

    def test_preset_maps_statistics_to_new_log_without_file_identity(self):
        workspace = Workspace(documents=[self.doc])
        workspace.panels[0].statistics[self.doc.id+"|rpm"] = ["mean"]
        workspace.panels[0].bounds = (1, 2)
        preset = workspace.preset("比較")
        self.assertNotIn(self.doc.id, json.dumps(preset))
        self.assertNotIn(str(self.path), json.dumps(preset))
        self.assertIsNone(preset["panels"][0]["bounds"])
        new_doc = Document.open(self.path)
        other = Workspace(documents=[new_doc])
        other.apply_preset(preset)
        self.assertEqual(other.panels[0].statistics[new_doc.id+"|rpm"], ["mean"])
        preset["panels"][0]["parameters"].append("unknown")
        self.assertEqual(other.apply_preset(preset), ["unknown"])

    def test_export_retains_source_columns_and_recorded_timestamps(self):
        original = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.doc.anchor = .1
        destination = self.root / "segment.csv"
        export_segment(self.doc, (0, .31), destination, [self.doc])
        frame = pd.read_csv(destination)
        self.assertEqual(frame["Time (msec)"].tolist(), [1100,1200,1300,1400])
        self.assertEqual(frame.columns[1:].tolist(), self.doc.loaded.metadata.columns_original)
        self.assertEqual(len(frame), 4)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), original)

    def test_source_overwrite_is_rejected_even_through_symlink(self):
        alias = self.root / "alias.csv"
        alias.symlink_to(self.path)
        for path in (self.path, alias):
            with self.assertRaises(ValueError):
                export_segment(self.doc, (0,1), path, [self.doc])
            with self.assertRaises(ValueError):
                write_json(path, {}, [self.doc])
        self.assertEqual(self.path.read_text(), CSV)

    def test_invalid_ranges_formats_and_non_finite_values_are_rejected(self):
        for bounds in ((1,1), (2,1), (0,float("nan")), (float("inf"),2)):
            with self.assertRaises(ValueError):
                time_range(*bounds)
        with self.assertRaises(ValueError):
            Workspace.restore({"format":"romraider-workspace","version":99}, [])
        with self.assertRaises(ValueError):
            Panel.parse({"y_left":[0,float("inf")]})


if __name__ == "__main__":
    unittest.main()
