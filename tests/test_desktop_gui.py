"""Behavior tests run offscreen; install the desktop extra to enable them."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HAS_QT = importlib.util.find_spec("PySide6") is not None and importlib.util.find_spec("pyqtgraph") is not None
if HAS_QT:
    from PySide6 import QtWidgets
    from src.desktop.app import MainWindow
    from src.desktop.model import Document


@unittest.skipUnless(HAS_QT, "desktop extra is not installed")
class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.window = MainWindow(self.root / "presets")
        self.window.show()
        paths = []
        for name, offset in (("a",0),("b",500)):
            path = self.root / (name+".csv")
            path.write_text("Time (msec),Engine Speed (rpm),Manifold Relative Pressure (kPa),Mass Airflow (g/s)\n" +
                            "\n".join(f"{i*100},{1000+i*100+offset},{i-50},{i+10}" for i in range(101)))
            paths.append(path)
        self.window._loaded([Document.open(p) for p in paths], [])
        self.app.processEvents()

    def tearDown(self):
        self.window.dirty = False
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.tmp.cleanup()

    def test_numeric_and_overview_selection_synchronize_both_endpoints(self):
        w = self.window
        w.start.setValue(2)
        w.end.setValue(6)
        w.apply_bounds()
        self.assertEqual(w.workspace.bounds, (2,6))
        for chart in w.charts:
            self.assertAlmostEqual(chart.plot.viewRange()[0][0], 2)
            self.assertAlmostEqual(chart.plot.viewRange()[0][1], 6)
        w.overview.region.setRegion((3,7))
        self.assertEqual(w.workspace.bounds, (3,7))
        self.assertEqual(w.start.value(), 3)
        self.assertEqual(w.end.value(), 7)

    def test_alignment_changes_display_coordinates_not_source_data(self):
        w = self.window
        w.logs.setCurrentRow(1)
        w.anchor.setValue(4)
        w.align_log()
        self.assertEqual(w.workspace.documents[1].anchor, 4)
        self.assertEqual(w.workspace.documents[1].elapsed[0], 0)
        self.assertEqual(w.workspace.documents[1].x[0], -4)
        self.assertEqual(w.workspace.extent(), (-4,10))

    def test_unlinked_charts_keep_independent_ranges_after_rebuild(self):
        w = self.window
        w.linked.setChecked(False)
        w.set_bounds(2,4)
        w.select_panel(1)
        w.set_bounds(5,8)
        w.rebuild_charts()
        self.assertEqual(w.workspace.panels[0].bounds, (2,4))
        self.assertEqual(w.workspace.panels[1].bounds, (5,8))
        self.assertAlmostEqual(w.charts[0].plot.viewRange()[0][0], 2)
        self.assertAlmostEqual(w.charts[1].plot.viewRange()[0][0], 5)
        w.linked.setChecked(True)
        self.assertEqual(w.charts[0].plot.viewRange()[0], w.charts[1].plot.viewRange()[0])

    def test_statistics_line_updates_when_either_time_endpoint_changes(self):
        w = self.window
        w.set_bounds(2,4)
        w.stat_checks["mean"].setChecked(True)
        self.assertEqual(len(w.charts[0].lines), 1)
        self.assertAlmostEqual(w.charts[0].lines[0][1].value(), 4000)
        w.set_bounds(5,7)
        self.assertAlmostEqual(w.charts[0].lines[0][1].value(), 7000)
        w.stat_scope.setCurrentIndex(1)
        self.assertAlmostEqual(w.charts[0].lines[0][1].value(), 6000)

    def test_cursor_values_and_panel_reorder(self):
        w = self.window
        w.set_cursor(0, 2)
        w.set_cursor(1, 3)
        self.assertEqual(w.values.item(0,1).text(), "3000")
        self.assertEqual(w.values.item(0,3).text(), "1000")
        w.move_panel(0,1)
        self.assertEqual(w.workspace.panels[1].parameters, ["rpm"])
        w.remove_panel(0)
        self.assertEqual(len(w.charts), 1)

    def test_session_save_and_open_restore_actual_window_controls(self):
        w = self.window
        w.logs.setCurrentRow(1)
        w.anchor.setValue(3)
        w.align_log()
        w.set_bounds(-1,4)
        w.set_cursor(0,2)
        w.stat_checks["median"].setChecked(True)
        destination = self.root / "workspace.json"
        with patch("src.desktop.app.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(destination),"")):
            self.assertTrue(w.save_session())
        w.set_bounds(0,1)
        with patch.object(w, "_maybe_save", return_value=True), patch("src.desktop.app.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(destination),"")):
            w.open_session()
        self.assertFalse(w.dirty)
        self.assertEqual(w.workspace.bounds, (-1,4))
        self.assertEqual(w.start.value(), -1)
        self.assertEqual(w.end.value(), 4)
        self.assertEqual(w.workspace.documents[1].anchor, 3)
        self.assertTrue(w.stat_checks["median"].isChecked())
        self.assertEqual(w.cursor_inputs[0].value(), 2)

    def test_saved_unlinked_panel_selection_matches_restored_range_controls(self):
        w = self.window
        w.linked.setChecked(False)
        w.set_bounds(1,3)
        w.select_panel(1)
        w.set_bounds(6,8)
        destination = self.root / "unlinked.json"
        with patch("src.desktop.app.QtWidgets.QFileDialog.getSaveFileName", return_value=(str(destination),"")):
            w.save_session()
        w.select_panel(0)
        with patch.object(w, "_maybe_save", return_value=True), patch("src.desktop.app.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(destination),"")):
            w.open_session()
        self.assertEqual(w.active, 1)
        self.assertEqual(w.start.value(), 6)
        self.assertEqual(w.end.value(), 8)
        self.assertEqual(w.workspace.panels[0].bounds, (1,3))

    def test_background_csv_loading_finishes_and_updates_ui(self):
        from PySide6 import QtCore
        path = self.root / "c.csv"
        path.write_bytes((self.root / "a.csv").read_bytes())
        w = self.window
        w.load_paths([str(path)])
        loop = QtCore.QEventLoop()
        w._load_task.signals.done.connect(loop.quit)
        QtCore.QTimer.singleShot(5000, loop.quit)
        loop.exec()
        self.assertFalse(w.loading)
        self.assertEqual(len(w.workspace.documents), 3)
        self.assertEqual(w.logs.count(), 3)

    def test_time_zoom_keeps_vertical_mouse_zoom_disabled(self):
        for chart in self.window.charts:
            self.assertEqual(chart.plot.getViewBox().state["mouseEnabled"], [True, False])


if __name__ == "__main__":
    unittest.main()
