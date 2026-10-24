"""Qt/pyqtgraph display components; statistics always come from source arrays."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from .model import LABELS, STAT_LABELS, Panel, Workspace


class ChartPanel(QtWidgets.QFrame):
    selected = QtCore.Signal(int)
    range_changed = QtCore.Signal(int, float, float)
    cursor_changed = QtCore.Signal(int, float)
    move_requested = QtCore.Signal(int, int)
    remove_requested = QtCore.Signal(int)
    edited = QtCore.Signal()

    def __init__(self, workspace: Workspace, config: Panel, index: int, parent=None):
        super().__init__(parent)
        self.workspace, self.config, self.index = workspace, config, index
        self.setObjectName("chartPanel")
        self.setMinimumHeight(280)
        self._updating = True
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 6)
        toolbar = QtWidgets.QHBoxLayout()
        self.select_button = QtWidgets.QPushButton(f"グラフ {index + 1}")
        self.select_button.setCheckable(True)
        self.select_button.clicked.connect(lambda: self.selected.emit(index))
        toolbar.addWidget(self.select_button)
        title = QtWidgets.QLineEdit(config.title)
        title.setPlaceholderText("グラフ名")
        title.editingFinished.connect(lambda: self._rename(title.text()))
        toolbar.addWidget(title, 1)
        for label, direction in (("↑", -1), ("↓", 1)):
            b = QtWidgets.QPushButton(label)
            b.setToolTip("グラフを並べ替える")
            b.setFixedWidth(30)
            b.clicked.connect(lambda checked=False, d=direction: self.move_requested.emit(index, d))
            toolbar.addWidget(b)
        remove = QtWidgets.QPushButton("削除")
        remove.clicked.connect(lambda: self.remove_requested.emit(index))
        toolbar.addWidget(remove)
        layout.addLayout(toolbar)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(205)
        self.plot.setLabel("bottom", "比較時刻", units="s")
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.showGrid(x=True, y=True, alpha=.13)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.plot.getPlotItem().hideButtons()
        layout.addWidget(self.plot)
        self.legend = self.plot.addLegend(offset=(60, 8), labelTextSize="10pt")
        self.right = pg.ViewBox()
        self.plot.scene().addItem(self.right)
        self.plot.showAxis("right")
        self.plot.getAxis("right").linkToView(self.right)
        self.right.setXLink(self.plot.getViewBox())
        self.plot.getViewBox().sigResized.connect(self._right_geometry)
        self.lines = []
        self.cursor_lines = []
        self._render()
        self.plot.getViewBox().sigXRangeChanged.connect(self._range)
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self._updating = False

    def _rename(self, value):
        if value.strip() and value != self.config.title:
            self.config.title = value.strip()
            self.edited.emit()

    def _right_geometry(self):
        self.right.setGeometry(self.plot.getViewBox().sceneBoundingRect())
        self.right.linkedViewChanged(self.plot.getViewBox(), self.right.XAxis)

    def _render(self):
        left_units, right_units = set(), set()
        missing = []
        enabled = [d for d in self.workspace.documents if d.enabled]
        for parameter in self.config.parameters:
            label = LABELS.get(parameter, parameter)
            found = False
            for doc_index, doc in enumerate(self.workspace.documents):
                if not doc.enabled:
                    continue
                if parameter not in doc.columns:
                    missing.append(f"{doc.name}：{label}")
                    continue
                found = True
                on_right = parameter in self.config.right
                target = self.right if on_right else self.plot
                (right_units if on_right else left_units).add(doc.units[parameter])
                x, y = doc.plot_arrays(parameter)
                style = [QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenStyle.DashLine,
                         QtCore.Qt.PenStyle.DotLine, QtCore.Qt.PenStyle.DashDotLine][doc_index % 4]
                pen = pg.mkPen(self.config.color(parameter), width=1.8, style=style)
                item = pg.PlotDataItem(x, y, pen=pen, connect="finite", name=f"{doc.name} · {label}")
                # Clip/downsample only chronological logs. Never hide clock resets by sorting.
                chronological = np.all(np.diff(doc.elapsed[np.isfinite(doc.elapsed)]) > 0)
                target.addItem(item)
                if chronological:
                    item.setDownsampling(auto=True, method="peak")
                    item.setClipToView(True)
                if on_right:
                    self.legend.addItem(item, f"{doc.name} · {label}（右軸）")
                key = doc.id + "|" + parameter
                stats = doc.statistics(parameter, self.config.bounds or self.workspace.bounds)
                for statistic in self.config.statistics.get(key, []):
                    value = stats.get(statistic)
                    if value is None:
                        continue
                    line = pg.InfiniteLine(pos=value, angle=0, movable=False,
                        pen=pg.mkPen(self.config.color(parameter), width=1,
                                     style=QtCore.Qt.PenStyle.DotLine),
                        label=f"{doc.name} {label} {STAT_LABELS[statistic]} {value:.3g}",
                        labelOpts={"position": .98, "anchors": [(1, 0), (1, 1)],
                                   "color": self.config.color(parameter)})
                    target.addItem(line)
                    self.lines.append((target, line, doc, parameter, statistic))
            if not found and not enabled:
                missing.append(label)
        self.plot.setLabel("left", " / ".join(sorted(left_units - {""})) or "記録値")
        self.plot.setLabel("right", " / ".join(sorted(right_units - {""})) or "")
        self.plot.showAxis("right", bool(right_units))
        if missing:
            self.setToolTip("存在しない項目：\n" + "\n".join(missing))
        if not enabled:
            label = pg.TextItem("CSVを追加して、左側から表示項目を選択してください", color="#78879b")
            self.plot.addItem(label)
        for i, value in enumerate(self.workspace.cursors):
            line = pg.InfiniteLine(value, angle=90, movable=True,
                pen=pg.mkPen("#718096", width=1, style=QtCore.Qt.PenStyle.DashLine),
                label=f"C{i+1}", labelOpts={"position": .91-i*.10, "color": "#718096"})
            line.sigPositionChanged.connect(lambda item, n=i: self._cursor(n, item.value()))
            self.plot.addItem(line)
            self.cursor_lines.append(line)
        for annotation in self.workspace.annotations:
            if annotation.start == annotation.end:
                mark = pg.InfiniteLine(annotation.start, angle=90, pen=pg.mkPen("#ad8f42", width=1))
            else:
                mark = pg.LinearRegionItem((annotation.start, annotation.end), movable=False,
                                          brush=pg.mkBrush(173, 143, 66, 20),
                                          pen=pg.mkPen(173, 143, 66, 75))
            self.plot.addItem(mark)
        self.set_range(*(self.config.bounds or self.workspace.bounds))
        for view, bounds in ((self.plot.getViewBox(), self.config.y_left), (self.right, self.config.y_right)):
            if bounds:
                view.setYRange(*bounds, padding=0)
            else:
                view.enableAutoRange(axis=pg.ViewBox.YAxis)
                view.setAutoVisible(y=True)
        self._right_geometry()

    def update_statistics(self, bounds):
        for target, line, doc, parameter, statistic in self.lines:
            value = doc.statistics(parameter, bounds)[statistic]
            line.setVisible(value is not None)
            if value is not None:
                line.setValue(value)
                line.label.setFormat(f"{doc.name} {LABELS.get(parameter, parameter)} {STAT_LABELS[statistic]} {value:.3g}")

    def _range(self, _, bounds):
        if not self._updating:
            self.range_changed.emit(self.index, float(bounds[0]), float(bounds[1]))

    def _clicked(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if self.plot.getViewBox().sceneBoundingRect().contains(event.scenePos()):
            self.selected.emit(self.index)
            x = self.plot.getViewBox().mapSceneToView(event.scenePos()).x()
            cursor = 1 if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier else 0
            self.cursor_changed.emit(cursor, x)

    def _cursor(self, index, value):
        if not self._updating:
            self.cursor_changed.emit(index, value)

    def set_range(self, start, end):
        previous = self._updating
        self._updating = True
        self.plot.setXRange(start, end, padding=0)
        self._updating = previous

    def set_cursor(self, index, value):
        previous = self._updating
        self._updating = True
        self.cursor_lines[index].setValue(value)
        self._updating = previous

    def dispose(self):
        self._updating = True
        self.hide()
        self.plot.scene().removeItem(self.right)
        self.plot.close()
        self.deleteLater()


class Overview(pg.PlotWidget):
    selected = QtCore.Signal(float, float)
    selection_started = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(95)
        self.hideAxis("left")
        self.setLabel("bottom", "比較時刻", units="s")
        self.getAxis("bottom").enableAutoSIPrefix(False)
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.region = None

    def rebuild(self, workspace):
        self.clear()
        for i, doc in enumerate(workspace.documents):
            if not doc.enabled:
                continue
            parameter = "rpm" if "rpm" in doc.columns else next(iter(doc.columns), None)
            if parameter:
                x, y = doc.plot_arrays(parameter)
                style = QtCore.Qt.PenStyle.SolidLine if i == 0 else QtCore.Qt.PenStyle.DashLine
                self.plot(x, y, pen=pg.mkPen("#8290a3", width=1, style=style), connect="finite")
        self.setXRange(*workspace.extent(), padding=.01)
        self.region = pg.LinearRegionItem(workspace.bounds, bounds=workspace.extent(),
                                          brush=pg.mkBrush(57, 123, 212, 35),
                                          pen=pg.mkPen("#397bd4", width=1))
        self.addItem(self.region)
        self._selecting = False
        self.region.sigRegionChanged.connect(self._region_changed)
        self.region.sigRegionChangeFinished.connect(self._region_finished)

    def _region_changed(self):
        if not self._selecting:
            self._selecting = True
            self.selection_started.emit()
        self.selected.emit(*self.region.getRegion())

    def _region_finished(self):
        self._selecting = False

    def set_region(self, start, end):
        if self.region:
            blocker = QtCore.QSignalBlocker(self.region)
            self.region.setRegion((start, end))
            del blocker
