"""RomRaider desktop application. Run with ``python -m src.desktop``."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from uuid import uuid4

import pyqtgraph as pg
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .charts import ChartPanel, Overview
from .model import (Annotation, Document, LABELS, Panel, STAT_LABELS, Workspace,
                    export_segment, finite, protect_sources, time_range,
                    validate_format, write_json)


def number(decimals=3):
    widget = QtWidgets.QDoubleSpinBox()
    widget.setRange(-1e9, 1e9)
    widget.setDecimals(decimals)
    widget.setKeyboardTracking(False)
    widget.setSuffix(" 秒")
    return widget


def read_json(path: Path) -> dict:
    if path.stat().st_size > 2_000_000:
        raise ValueError("設定ファイルが大きすぎます（上限2MB）。")
    return json.loads(path.read_text(encoding="utf-8"))


class LoadSignals(QtCore.QObject):
    done = QtCore.Signal(object, object)


class LoadTask(QtCore.QRunnable):
    def __init__(self, paths):
        super().__init__()
        self.paths = paths
        self.signals = LoadSignals()

    @QtCore.Slot()
    def run(self):
        documents, errors = [], []
        for path in self.paths:
            try:
                documents.append(Document.open(Path(path)))
            except Exception as exc:
                errors.append(f"{Path(path).name}：{exc}")
        self.signals.done.emit(documents, errors)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, preset_directory: Path | None = None):
        super().__init__()
        self.workspace = Workspace()
        self.active = 0
        self.charts = []
        self.history = []
        self.dirty = False
        self.session_path = None
        self.loading = False
        self._sync = False
        self._rebuilding = False
        self._load_task = None
        self.preset_directory = preset_directory or Path(__file__).resolve().parents[2] / ".local" / "desktop" / "presets"
        self.resize(1440, 960)
        self.setMinimumSize(1000, 720)
        self.setAcceptDrops(True)
        self._build()
        self._refresh()
        self._title()

    def _title(self):
        name = self.session_path.stem if self.session_path else "新しい分析"
        self.setWindowTitle(f"{'● ' if self.dirty else ''}{name} — RomRaider Log Analyzer")

    @property
    def active(self):
        return self.workspace.active_panel

    @active.setter
    def active(self, value):
        self.workspace.active_panel = value

    def changed(self):
        self.dirty = True
        self._title()

    def error(self, message):
        QtWidgets.QMessageBox.warning(self, "操作を完了できませんでした", str(message))

    def _button(self, text, callback, layout=None):
        b = QtWidgets.QPushButton(text)
        b.clicked.connect(callback)
        if layout is not None:
            layout.addWidget(b)
        return b

    def _build(self):
        toolbar = self.addToolBar("ファイルと表示")
        toolbar.setMovable(False)
        for text, callback, shortcut in (
            ("＋ ログを追加", self.choose_logs, "Ctrl+O"),
            ("分析を開く", self.open_session, "Ctrl+Shift+O"),
            ("分析を保存", self.save_session, "Ctrl+S"),
            ("プリセット", self.presets, None),
            ("画像を出力", self.export_image, None),
            ("区間CSVを出力", self.export_csv, None),
        ):
            action = QtGui.QAction(text, self)
            action.triggered.connect(callback)
            if shortcut:
                action.setShortcut(QtGui.QKeySequence(shortcut))
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel(" 配置 "))
        self.columns = QtWidgets.QComboBox()
        self.columns.addItems(["縦に並べる", "2列に並べる"])
        self.columns.currentIndexChanged.connect(self.layout_changed)
        toolbar.addWidget(self.columns)
        self.linked = QtWidgets.QCheckBox("時間軸を連動")
        self.linked.setChecked(True)
        self.linked.toggled.connect(self.link_changed)
        toolbar.addWidget(self.linked)
        self.toggle_inspector = QtGui.QAction("設定欄", self)
        self.toggle_inspector.setCheckable(True)
        self.toggle_inspector.setChecked(True)
        self.toggle_inspector.toggled.connect(lambda show: self.inspector.setVisible(show))
        toolbar.addAction(self.toggle_inspector)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(10, 8, 10, 8)
        self.splitter = QtWidgets.QSplitter()
        outer.addWidget(self.splitter, 1)

        side = QtWidgets.QWidget()
        side.setMinimumWidth(190)
        side.setMaximumWidth(340)
        sl = QtWidgets.QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 8, 0)
        sl.addWidget(QtWidgets.QLabel("ログ（チェックで表示切り替え）"))
        self.logs = QtWidgets.QListWidget()
        self.logs.setMaximumHeight(180)
        self.logs.currentRowChanged.connect(self.select_log)
        self.logs.itemChanged.connect(self.log_changed)
        self.logs.itemDoubleClicked.connect(self.rename_log)
        sl.addWidget(self.logs)
        row = QtWidgets.QHBoxLayout()
        self._button("詳細", self.log_details, row)
        self._button("削除", self.remove_log, row)
        sl.addLayout(row)
        sl.addWidget(QtWidgets.QLabel("比較の基準点（元ログの経過秒）"))
        self.anchor = number()
        self.anchor.setMinimum(0)
        sl.addWidget(self.anchor)
        self._button("この時刻を0秒にする", self.align_log, sl)
        self._button("C1位置を基準点にする", self.align_cursor, sl)
        sl.addWidget(QtWidgets.QLabel("表示する項目 · 選択中のグラフ"))
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("項目名を検索")
        self.search.textChanged.connect(self.filter_parameters)
        sl.addWidget(self.search)
        self.parameters = QtWidgets.QListWidget()
        self.parameters.itemChanged.connect(self.parameter_changed)
        sl.addWidget(self.parameters, 1)
        self._button("＋ 新しいグラフ", self.add_panel, sl)
        self.splitter.addWidget(side)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.chart_container = QtWidgets.QWidget()
        self.chart_layout = QtWidgets.QGridLayout(self.chart_container)
        self.chart_layout.setContentsMargins(2, 0, 2, 0)
        self.chart_layout.setSpacing(10)
        self.scroll.setWidget(self.chart_container)
        self.splitter.addWidget(self.scroll)

        self.inspector = QtWidgets.QScrollArea()
        self.inspector.setWidgetResizable(True)
        self.inspector.setMinimumWidth(270)
        self.inspector.setMaximumWidth(400)
        self.inspector.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        contents = QtWidgets.QWidget()
        il = QtWidgets.QVBoxLayout(contents)
        il.setContentsMargins(10, 0, 0, 0)
        il.addWidget(QtWidgets.QLabel("系列の設定・統計"))
        self.series = QtWidgets.QComboBox()
        self.series.currentIndexChanged.connect(self.series_changed)
        il.addWidget(self.series)
        row = QtWidgets.QHBoxLayout()
        self.axis = QtWidgets.QComboBox()
        self.axis.addItems(["左軸", "右軸"])
        self.axis.currentIndexChanged.connect(self.axis_changed)
        row.addWidget(self.axis)
        self._button("線の色", self.choose_color, row)
        il.addLayout(row)
        self.stat_scope = QtWidgets.QComboBox()
        self.stat_scope.addItems(["統計対象：選択区間", "統計対象：ログ全体"])
        self.stat_scope.currentIndexChanged.connect(self.stat_scope_changed)
        il.addWidget(self.stat_scope)
        self.stat_labels, self.stat_checks = {}, {}
        for key, label in STAT_LABELS.items():
            row = QtWidgets.QHBoxLayout()
            check = QtWidgets.QCheckBox(label + "線")
            check.toggled.connect(lambda value, stat=key: self.stat_changed(stat, value))
            row.addWidget(check)
            output = QtWidgets.QLabel("—")
            output.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            row.addWidget(output, 1)
            il.addLayout(row)
            self.stat_checks[key], self.stat_labels[key] = check, output
        self.count = QtWidgets.QLabel("有効データ数：—")
        il.addWidget(self.count)
        self._button("縦軸の範囲を設定", self.y_limits, il)
        self._button("縦軸を自動調整", self.y_auto, il)
        il.addSpacing(14)
        il.addWidget(QtWidgets.QLabel("カーソル値 · クリック=C1 / Shift+クリック=C2"))
        self.cursor_inputs = []
        for i in range(2):
            cursor_row = QtWidgets.QHBoxLayout()
            cursor_row.addWidget(QtWidgets.QLabel(f"C{i+1}"))
            control = number()
            control.valueChanged.connect(lambda value, n=i: self.set_cursor(n, value))
            self.cursor_inputs.append(control)
            cursor_row.addWidget(control)
            il.addLayout(cursor_row)
        self.delta = QtWidgets.QLabel("時間差：1.000 秒")
        il.addWidget(self.delta)
        self.values = QtWidgets.QTableWidget(0, 4)
        self.values.setHorizontalHeaderLabels(["系列", "C1", "C2", "差"])
        self.values.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.values.verticalHeader().hide()
        self.values.setMinimumHeight(150)
        self.values.setMaximumHeight(240)
        self.values.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        il.addWidget(self.values)
        il.addWidget(QtWidgets.QLabel("注釈（比較時刻で保存）"))
        self.annotations = QtWidgets.QListWidget()
        self.annotations.setMinimumHeight(80)
        self.annotations.setMaximumHeight(140)
        self.annotations.itemClicked.connect(self.jump_annotation)
        il.addWidget(self.annotations)
        row = QtWidgets.QHBoxLayout()
        self._button("＋ 追加", self.add_annotation, row)
        self._button("編集", self.edit_annotation, row)
        self._button("削除", self.remove_annotation, row)
        il.addLayout(row)
        il.addStretch()
        for label in contents.findChildren(QtWidgets.QLabel):
            label.setWordWrap(True)
        self.inspector.setWidget(contents)
        self.splitter.addWidget(self.inspector)
        self.splitter.setSizes([215, 850, 320])

        self.overview = Overview()
        self.overview.selected.connect(lambda start, end: self.set_bounds(start, end, remember=False))
        self.overview.selection_started.connect(self.remember_range)
        outer.addWidget(self.overview)
        row = QtWidgets.QHBoxLayout()
        self.start, self.end = number(), number()
        row.addWidget(QtWidgets.QLabel("開始"))
        row.addWidget(self.start)
        row.addWidget(QtWidgets.QLabel("終了"))
        row.addWidget(self.end)
        self._button("範囲を適用", self.apply_bounds, row)
        self._button("全体表示", lambda: self.set_bounds(*self.workspace.extent()), row)
        self._button("前の範囲", self.previous_range, row)
        row.addStretch()
        self.range_label = QtWidgets.QLabel()
        row.addWidget(self.range_label)
        outer.addLayout(row)
        self.statusBar().showMessage("CSVを追加するか、ウィンドウへドラッグしてください。元ログは変更しません。")
        menu = self.menuBar().addMenu("ファイル")
        for action in toolbar.actions()[:6]:
            menu.addAction(action)
        save_as = menu.addAction("分析を別名で保存…")
        save_as.triggered.connect(lambda: self.save_session(save_as=True))

    def _refresh(self):
        self._sync = True
        old_log = self.logs.currentRow()
        self.logs.clear()
        for d in self.workspace.documents:
            item = QtWidgets.QListWidgetItem(d.name + "\n" + d.path.name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if d.enabled else QtCore.Qt.CheckState.Unchecked)
            item.setToolTip(str(d.path) + "\n" + "\n".join(d.warnings()))
            self.logs.addItem(item)
        self.logs.setCurrentRow(max(0, min(old_log, len(self.workspace.documents)-1)))
        self.columns.setCurrentIndex(self.workspace.columns-1)
        self.linked.setChecked(self.workspace.linked)
        self.stat_scope.setCurrentIndex(int(self.workspace.stat_all))
        self.start.setValue(self.workspace.bounds[0])
        self.end.setValue(self.workspace.bounds[1])
        for i, value in enumerate(self.workspace.cursors):
            self.cursor_inputs[i].setValue(value)
        self._sync = False
        self.active = max(0, min(self.active, len(self.workspace.panels)-1))
        self.refresh_parameters()
        self.rebuild_charts()
        self.select_log(self.logs.currentRow())
        self.refresh_annotations()
        self.update_values()

    def refresh_parameters(self):
        self._sync = True
        self.parameters.clear()
        keys = sorted({key for d in self.workspace.documents for key in d.columns}, key=lambda k: LABELS.get(k,k))
        panel = self.workspace.panels[self.active]
        keys += [k for k in panel.parameters if k not in keys]
        for key in keys:
            units = sorted({d.units[key] for d in self.workspace.documents if key in d.units} - {""})
            text = LABELS.get(key, key) + (" [" + "/".join(units) + "]" if units else "")
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if key in panel.parameters else QtCore.Qt.CheckState.Unchecked)
            original = [f"{d.name}: {d.columns[key]}" for d in self.workspace.documents if key in d.columns]
            item.setToolTip("\n".join(original) or "この項目は読み込んだログに存在しません。")
            self.parameters.addItem(item)
        self._sync = False
        self.filter_parameters(self.search.text())

    def rebuild_charts(self):
        self._rebuilding = True
        for chart in self.charts:
            self.chart_layout.removeWidget(chart)
            chart.dispose()
        self.charts = []
        for i, panel in enumerate(self.workspace.panels):
            chart = ChartPanel(self.workspace, panel, i)
            chart.selected.connect(self.select_panel)
            chart.range_changed.connect(self.chart_range)
            chart.cursor_changed.connect(self.set_cursor)
            chart.move_requested.connect(self.move_panel)
            chart.remove_requested.connect(self.remove_panel)
            chart.edited.connect(self.changed)
            self.chart_layout.addWidget(chart, i // self.workspace.columns, i % self.workspace.columns)
            chart.select_button.setChecked(i == self.active)
            self.charts.append(chart)
        self.overview.rebuild(self.workspace)
        self._rebuilding = False
        self.refresh_series()
        self.update_statistics()

    def choose_logs(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "RomRaider CSVを追加", "", "CSV (*.csv);;すべてのファイル (*)")
        if paths:
            self.load_paths(paths)

    def load_paths(self, paths):
        if self.loading:
            return
        paths = [p for p in paths if Path(p).resolve() not in {d.path for d in self.workspace.documents}]
        if not paths:
            self.statusBar().showMessage("選択したログはすでに読み込まれています。")
            return
        self.loading = True
        self.statusBar().showMessage("CSVを読み込んでいます…")
        self._load_task = LoadTask(paths)
        self._load_task.signals.done.connect(self._loaded)
        QtCore.QThreadPool.globalInstance().start(self._load_task)

    def _loaded(self, documents, errors):
        self.loading = False
        self._load_task = None
        first = not self.workspace.documents
        for doc in documents:
            if doc.name == doc.path.stem:
                index = len(self.workspace.documents)
                doc.name = "ログ" + (chr(65+index) if index < 26 else str(index+1))
            self.workspace.add(doc)
        if documents:
            if first:
                self.workspace.bounds = self.workspace.extent()
                self.workspace.cursors = [self.workspace.bounds[0], self.workspace.bounds[0]+1]
            self._refresh()
            self.changed()
        warnings = [f"{d.name}：" + " / ".join(d.warnings()) for d in documents if d.warnings()]
        self.statusBar().showMessage(f"{len(documents)}件を追加しました。" + (" 詳細に読み込み上の注意があります。" if warnings else ""))
        if errors:
            self.error("\n".join(errors))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(u.toLocalFile().lower().endswith(".csv") for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.load_paths([u.toLocalFile() for u in event.mimeData().urls() if u.toLocalFile().lower().endswith(".csv")])
        event.acceptProposedAction()

    def select_log(self, index):
        if self._sync:
            return
        valid = 0 <= index < len(self.workspace.documents)
        self.anchor.setEnabled(valid)
        if valid:
            self.anchor.setValue(self.workspace.documents[index].anchor)

    def log_changed(self, item):
        if self._sync:
            return
        self.workspace.documents[self.logs.row(item)].enabled = item.checkState() == QtCore.Qt.CheckState.Checked
        self.rebuild_charts()
        self.update_values()
        self.changed()

    def rename_log(self, item):
        doc = self.workspace.documents[self.logs.row(item)]
        text, ok = QtWidgets.QInputDialog.getText(self, "ログ名", "表示名", text=doc.name)
        if ok and text.strip():
            doc.name = text.strip()
            self._refresh()
            self.changed()

    def remove_log(self):
        index = self.logs.currentRow()
        if index >= 0:
            doc = self.workspace.documents.pop(index)
            for panel in self.workspace.panels:
                panel.statistics = {k:v for k,v in panel.statistics.items() if not k.startswith(doc.id+"|")}
            self._refresh()
            self.changed()

    def log_details(self):
        index = self.logs.currentRow()
        if index < 0:
            return
        doc = self.workspace.documents[index]
        m = doc.loaded.metadata
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("ログの詳細")
        dialog.resize(700, 500)
        layout = QtWidgets.QVBoxLayout(dialog)
        info = QtWidgets.QPlainTextEdit()
        info.setReadOnly(True)
        info.setPlainText(f"{doc.path}\n\n行数：{m.row_count:,}\n標準記録周期：{m.median_period_s:.4f} 秒\n"
                          + "\n".join(doc.warnings() or ["読み込み上の注意はありません。"])
                          + "\n\n記録項目\n" + "\n".join(m.columns_original))
        layout.addWidget(info)
        self._button("閉じる", dialog.accept, layout)
        dialog.exec()

    def align_log(self):
        index = self.logs.currentRow()
        if index < 0:
            return
        doc = self.workspace.documents[index]
        anchor = self.anchor.value()
        if anchor > float(np.max(doc.elapsed[np.isfinite(doc.elapsed)])):
            self.error("基準点はログの記録範囲内で指定してください。")
            return
        doc.anchor = anchor
        self.workspace.bounds = self.workspace.extent()
        for panel in self.workspace.panels:
            panel.bounds = None
        self.history.clear()
        self._refresh()
        self.changed()

    def align_cursor(self):
        index = self.logs.currentRow()
        if index >= 0:
            doc = self.workspace.documents[index]
            self.anchor.setValue(self.workspace.cursors[0] + doc.anchor)
            self.align_log()

    def filter_parameters(self, text):
        for i in range(self.parameters.count()):
            item = self.parameters.item(i)
            item.setHidden(text.casefold() not in (item.text()+item.toolTip()).casefold())

    def parameter_changed(self, item):
        if self._sync:
            return
        key = item.data(QtCore.Qt.ItemDataRole.UserRole)
        panel = self.workspace.panels[self.active]
        if item.checkState() == QtCore.Qt.CheckState.Checked:
            if key not in panel.parameters:
                panel.parameters.append(key)
        elif key in panel.parameters:
            panel.parameters.remove(key)
        self.rebuild_charts()
        self.update_values()
        self.changed()

    def select_panel(self, index):
        if self._rebuilding or index == self.active:
            return
        self.active = index
        for i, chart in enumerate(self.charts):
            chart.select_button.setChecked(i == index)
        panel = self.workspace.panels[index]
        if not self.workspace.linked:
            self.set_bounds(*(panel.bounds or self.workspace.bounds), remember=False)
        self.refresh_parameters()
        self.refresh_series()
        self.update_values()

    def add_panel(self):
        self.workspace.panels.append(Panel(f"グラフ {len(self.workspace.panels)+1}"))
        self.active = len(self.workspace.panels)-1
        self.refresh_parameters()
        self.rebuild_charts()
        self.changed()

    def remove_panel(self, index):
        if len(self.workspace.panels) == 1:
            self.error("グラフは1つ以上残してください。")
            return
        self.workspace.panels.pop(index)
        self.active = min(self.active, len(self.workspace.panels)-1)
        self.refresh_parameters()
        self.rebuild_charts()
        self.changed()

    def move_panel(self, index, direction):
        other = index + direction
        if 0 <= other < len(self.workspace.panels):
            self.workspace.panels[index], self.workspace.panels[other] = self.workspace.panels[other], self.workspace.panels[index]
            self.active = other
            self.refresh_parameters()
            self.rebuild_charts()
            self.changed()

    def layout_changed(self, index):
        if not self._sync:
            self.workspace.columns = index+1
            self.rebuild_charts()
            self.changed()

    def link_changed(self, enabled):
        if self._sync:
            return
        self.workspace.linked = enabled
        if enabled:
            for panel in self.workspace.panels:
                panel.bounds = None
        else:
            for panel in self.workspace.panels:
                panel.bounds = self.workspace.bounds
        self.rebuild_charts()
        self.changed()

    def apply_bounds(self):
        try:
            self.set_bounds(*time_range(self.start.value(), self.end.value()))
        except ValueError as exc:
            self.error(exc)

    def set_bounds(self, start, end, remember=True):
        if self._sync or self._rebuilding or not math.isfinite(start+end) or end <= start:
            return
        bounds = (float(start), float(end))
        previous = self.workspace.bounds
        if remember and bounds != previous:
            self.history.append((self.active, previous))
            self.history = self.history[-100:]
        self.workspace.bounds = bounds
        self._sync = True
        self.start.setValue(start)
        self.end.setValue(end)
        if self.workspace.linked:
            for chart in self.charts:
                chart.config.bounds = None
                chart.set_range(start, end)
        elif self.charts:
            self.workspace.panels[self.active].bounds = bounds
            self.charts[self.active].set_range(start, end)
        self.overview.set_region(start, end)
        self._sync = False
        self.range_label.setText(f"区間幅 {end-start:.3f} 秒")
        self.update_statistics()
        if bounds != previous:
            self.changed()

    def chart_range(self, index, start, end):
        if self._sync or self._rebuilding:
            return
        if self.active != index:
            self.select_panel(index)
        self.set_bounds(start, end)

    def previous_range(self):
        if self.history:
            index, bounds = self.history.pop()
            if index < len(self.workspace.panels):
                self.select_panel(index)
                self.set_bounds(*bounds, remember=False)

    def remember_range(self):
        self.history.append((self.active, self.workspace.bounds))
        self.history = self.history[-100:]

    def refresh_series(self):
        old = self.series.currentData()
        self._sync = True
        self.series.clear()
        panel = self.workspace.panels[self.active]
        for d in self.workspace.documents:
            if d.enabled:
                for key in panel.parameters:
                    if key in d.columns:
                        self.series.addItem(f"{d.name} · {LABELS.get(key,key)}", (d.id, key))
        index = self.series.findData(old)
        self.series.setCurrentIndex(max(0, index))
        self._sync = False
        self.series_changed()

    def current_series(self):
        selected = self.series.currentData()
        if selected:
            return next((d for d in self.workspace.documents if d.id == selected[0]), None), selected[1]
        return None, None

    def series_changed(self, *_):
        if self._sync:
            return
        doc, key = self.current_series()
        panel = self.workspace.panels[self.active]
        self._sync = True
        self.axis.setEnabled(doc is not None)
        self.axis.setCurrentIndex(1 if key in panel.right else 0)
        for stat, check in self.stat_checks.items():
            check.setEnabled(doc is not None)
            check.setChecked(doc is not None and stat in panel.statistics.get(doc.id+"|"+key, []))
        self._sync = False
        self.update_statistics()

    def axis_changed(self, index):
        if self._sync:
            return
        doc, key = self.current_series()
        if doc:
            panel = self.workspace.panels[self.active]
            panel.right = [p for p in panel.right if p != key]
            if index:
                panel.right.append(key)
            self.rebuild_charts()
            self.changed()

    def choose_color(self):
        doc, key = self.current_series()
        if doc:
            panel = self.workspace.panels[self.active]
            color = QtWidgets.QColorDialog.getColor(QtGui.QColor(panel.color(key)), self, "項目の線の色")
            if color.isValid():
                panel.colors[key] = color.name()
                self.rebuild_charts()
                self.changed()

    def stat_changed(self, statistic, enabled):
        if self._sync:
            return
        doc, key = self.current_series()
        if not doc:
            return
        selected = doc.id+"|"+key
        panel = self.workspace.panels[self.active]
        values = [s for s in panel.statistics.get(selected, []) if s != statistic]
        if enabled:
            values.append(statistic)
        panel.statistics[selected] = values
        self.rebuild_charts()
        self.changed()

    def update_statistics(self, *_):
        if self._sync:
            return
        bounds = self.workspace.bounds if self.stat_scope.currentIndex() == 0 else None
        for chart in self.charts:
            chart_bounds = (chart.config.bounds or self.workspace.bounds) if bounds is not None else None
            chart.update_statistics(chart_bounds)
        doc, key = self.current_series()
        stats = doc.statistics(key, bounds) if doc else {}
        unit = doc.units[key] if doc else ""
        for statistic, label in self.stat_labels.items():
            value = stats.get(statistic)
            label.setText(f"{value:,.4g} {unit}" if value is not None else "—")
        self.count.setText(f"有効データ数：{stats.get('count', 0):,}")

    def stat_scope_changed(self):
        if not self._sync:
            self.workspace.stat_all = self.stat_scope.currentIndex() == 1
            self.update_statistics()
            self.changed()

    def y_limits(self):
        panel = self.workspace.panels[self.active]
        side = "y_right" if self.axis.currentIndex() else "y_left"
        current = getattr(panel, side) or (0, 100)
        text, ok = QtWidgets.QInputDialog.getText(self, "縦軸の範囲", "最小値, 最大値（例：0, 6000）", text=f"{current[0]:g}, {current[1]:g}")
        if ok:
            try:
                setattr(panel, side, time_range(*text.replace("，", ",").split(",")))
                self.rebuild_charts()
                self.changed()
            except (ValueError, TypeError):
                self.error("最小値と最大値をカンマ区切りで指定してください。")

    def y_auto(self):
        panel = self.workspace.panels[self.active]
        setattr(panel, "y_right" if self.axis.currentIndex() else "y_left", None)
        self.rebuild_charts()
        self.changed()

    def set_cursor(self, index, value):
        if self._sync:
            return
        self.workspace.cursors[index] = value
        self._sync = True
        self.cursor_inputs[index].setValue(value)
        for chart in self.charts:
            chart.set_cursor(index, value)
        self._sync = False
        self.update_values()
        self.changed()

    def update_values(self):
        self.delta.setText(f"時間差 C2−C1：{self.workspace.cursors[1]-self.workspace.cursors[0]:.3f} 秒")
        rows = [(doc, p) for doc in self.workspace.documents if doc.enabled
                for p in self.workspace.panels[self.active].parameters if p in doc.columns]
        self.values.setRowCount(len(rows))
        for i, (doc, key) in enumerate(rows):
            samples = [doc.sample(key, x) for x in self.workspace.cursors]
            difference = samples[1][1]-samples[0][1] if all(samples) else None
            entries = [doc.name+" · "+LABELS.get(key,key),
                       *(f"{s[1]:.4g}" if s else "—" for s in samples),
                       f"{difference:.4g}" if difference is not None else "—"]
            for j, value in enumerate(entries):
                item = QtWidgets.QTableWidgetItem(value)
                if j:
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                item.setToolTip(f"単位：{doc.units[key]}\n" + " / ".join(f"C{k+1} 元ログ {s[0]:.4f}秒" if s else f"C{k+1}：近傍の有効な記録なし" for k,s in enumerate(samples)))
                self.values.setItem(i, j, item)

    def refresh_annotations(self):
        self.annotations.clear()
        for a in self.workspace.annotations:
            item = QtWidgets.QListWidgetItem(f"{a.start:.3f}〜{a.end:.3f}秒 · {a.label}")
            item.setToolTip(a.note)
            self.annotations.addItem(item)

    def annotation_dialog(self, existing=None):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("注釈")
        layout = QtWidgets.QFormLayout(dialog)
        title = QtWidgets.QLineEdit(existing.label if existing else "")
        note = QtWidgets.QPlainTextEdit(existing.note if existing else "")
        start, end = number(), number()
        start.setValue(existing.start if existing else self.workspace.bounds[0])
        end.setValue(existing.end if existing else self.workspace.bounds[1])
        layout.addRow("名前", title)
        layout.addRow("開始（点の注釈は開始＝終了）", start)
        layout.addRow("終了", end)
        layout.addRow("メモ", note)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            if not title.text().strip() or end.value() < start.value():
                self.error("名前と有効な時間範囲を指定してください。")
                return None
            return Annotation(start.value(), end.value(), title.text().strip(), note.toPlainText())
        return None

    def add_annotation(self):
        annotation = self.annotation_dialog()
        if annotation:
            self.workspace.annotations.append(annotation)
            self.refresh_annotations()
            self.rebuild_charts()
            self.changed()

    def edit_annotation(self):
        row = self.annotations.currentRow()
        if row >= 0:
            annotation = self.annotation_dialog(self.workspace.annotations[row])
            if annotation:
                self.workspace.annotations[row] = annotation
                self.refresh_annotations()
                self.rebuild_charts()
                self.changed()

    def remove_annotation(self):
        row = self.annotations.currentRow()
        if row >= 0:
            self.workspace.annotations.pop(row)
            self.refresh_annotations()
            self.rebuild_charts()
            self.changed()

    def jump_annotation(self, item):
        annotation = self.workspace.annotations[self.annotations.row(item)]
        width = max(annotation.end-annotation.start, 1)
        self.set_bounds(annotation.start-width*.1, annotation.end+width*.1)

    def _maybe_save(self):
        if not self.dirty:
            return True
        answer = QtWidgets.QMessageBox.question(self, "分析状態の保存", "現在の分析を保存しますか？",
            QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Save)
        if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
            return False
        return self.save_session() if answer == QtWidgets.QMessageBox.StandardButton.Save else True

    def save_session(self, checked=False, save_as=False):
        path = None if save_as else self.session_path
        if path is None:
            selected, _ = QtWidgets.QFileDialog.getSaveFileName(self, "分析状態を保存", "analysis.rrworkspace.json", "分析状態 (*.json)")
            if not selected:
                return False
            path = Path(selected)
        try:
            write_json(path, self.workspace.session(path), self.workspace.documents)
            self.session_path, self.dirty = path, False
            self._title()
            self.statusBar().showMessage("分析状態を保存しました。元CSVは別に保管してください。")
            return True
        except (ValueError, OSError) as exc:
            self.error(exc)
            return False

    def open_session(self):
        if self.loading or not self._maybe_save():
            return
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(self, "分析状態を開く", "", "分析状態 (*.json)")
        if not selected:
            return
        path = Path(selected)
        try:
            data = read_json(path)
            validate_format(data, "romraider-workspace")
            documents = []
            for entry in data["documents"]:
                source = (path.parent / entry["path"]).resolve()
                if not source.is_file():
                    replacement, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"元ログを選び直してください：{source.name}", "", "CSV (*.csv)")
                    if not replacement:
                        return
                    source = Path(replacement)
                doc = Document.open(source)
                if entry.get("sha256") != doc.digest:
                    answer = QtWidgets.QMessageBox.question(self, "元ログの内容が変更されています", f"{source.name}\n保存時と異なる内容です。このログで復元しますか？")
                    if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                        return
                doc.id, doc.name = str(entry["id"]), str(entry["name"])
                doc.anchor, doc.enabled = finite(entry["anchor"]), bool(entry["enabled"])
                if any(d.id == doc.id for d in documents):
                    raise ValueError("保存されたログIDが重複しています。")
                documents.append(doc)
            workspace = Workspace.restore(data, documents)
            self.workspace, self.session_path = workspace, path
            self.history = []
            self._refresh()
            self.dirty = False
            self._title()
            self.statusBar().showMessage("分析状態を復元しました。")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.error(f"分析状態を開けませんでした：{exc}")

    def presets(self):
        dialog = PresetDialog(self)
        dialog.exec()

    def export_image(self):
        if not self.charts:
            return
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(self, "グラフ一覧を画像に保存", "graphs.png", "PNG (*.png)")
        if selected:
            try:
                protect_sources(Path(selected), self.workspace.documents)
                if not self.chart_container.grab().save(selected, "PNG"):
                    raise OSError("画像を書き込めませんでした。")
                self.statusBar().showMessage("グラフ一覧を画像として保存しました。")
            except (OSError, ValueError) as exc:
                self.error(exc)

    def export_csv(self):
        index = self.logs.currentRow()
        if index < 0:
            self.error("出力するログを左側で選択してください。")
            return
        doc = self.workspace.documents[index]
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(self, f"{doc.name} の選択区間を出力", f"{doc.path.stem}-segment.csv", "CSV (*.csv)")
        if selected:
            try:
                export_segment(doc, self.workspace.bounds, Path(selected), self.workspace.documents)
                self.statusBar().showMessage("選択ログの区間CSVを出力しました。先頭列は比較時刻です。")
            except (OSError, ValueError) as exc:
                self.error(exc)

    def closeEvent(self, event):
        if self.loading:
            self.statusBar().showMessage("CSVの読み込み完了後に閉じてください。")
            event.ignore()
        elif self._maybe_save():
            event.accept()
        else:
            event.ignore()


class PresetDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("プリセット")
        self.resize(560, 420)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("表示設定を別のログにも適用できます。適用後に調整し、保存して編集できます。"))
        self.items = QtWidgets.QListWidget()
        self.items.itemDoubleClicked.connect(self.apply)
        layout.addWidget(self.items)
        row = QtWidgets.QHBoxLayout()
        for text, callback in (("適用", self.apply), ("現在の設定を保存", self.save),
                               ("複製", self.duplicate), ("名前変更", self.rename), ("削除", self.delete)):
            window._button(text, callback, row)
        layout.addLayout(row)
        window._button("閉じる", self.accept, layout)
        self.refresh()

    def refresh(self):
        self.items.clear()
        for path in sorted(self.window.preset_directory.glob("*.json")):
            try:
                data = read_json(path)
                validate_format(data, "romraider-preset")
                item = QtWidgets.QListWidgetItem(str(data.get("name", path.stem)))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, str(path))
                self.items.addItem(item)
            except (ValueError, OSError):
                pass

    def current(self):
        item = self.items.currentItem()
        return Path(item.data(QtCore.Qt.ItemDataRole.UserRole)) if item else None

    def apply(self, *_):
        path = self.current()
        if path:
            try:
                missing = self.window.workspace.apply_preset(read_json(path))
                self.window.active = 0
                self.window._refresh()
                self.window.changed()
                if missing:
                    self.window.statusBar().showMessage("ログに存在しない項目：" + ", ".join(LABELS.get(k,k) for k in missing))
                self.accept()
            except (ValueError, OSError, KeyError, TypeError) as exc:
                self.window.error(exc)

    def save(self):
        path = self.current()
        name, ok = QtWidgets.QInputDialog.getText(self, "プリセットを保存", "名前（選択中と同じ名前なら上書き）", text=self.items.currentItem().text() if path else "")
        if ok and name.strip():
            name = name.strip()
            if not path or self.items.currentItem().text() != name:
                path = self.window.preset_directory / (uuid4().hex + ".json")
            try:
                self.window.preset_directory.mkdir(parents=True, exist_ok=True)
                write_json(path, self.window.workspace.preset(name), self.window.workspace.documents)
                self.refresh()
            except (OSError, ValueError) as exc:
                self.window.error(exc)

    def duplicate(self):
        path = self.current()
        if path:
            try:
                data = read_json(path)
                data["name"] = str(data["name"]) + " のコピー"
                write_json(path.with_name(uuid4().hex + ".json"), data)
                self.refresh()
            except (ValueError, OSError, KeyError) as exc:
                self.window.error(exc)

    def rename(self):
        path = self.current()
        if path:
            name, ok = QtWidgets.QInputDialog.getText(self, "名前変更", "名前", text=self.items.currentItem().text())
            if ok and name.strip():
                try:
                    data = read_json(path)
                    data["name"] = name.strip()
                    write_json(path, data)
                    self.refresh()
                except (ValueError, OSError) as exc:
                    self.window.error(exc)

    def delete(self):
        path = self.current()
        if path and QtWidgets.QMessageBox.question(self, "プリセットを削除", "選択したプリセットを削除しますか？") == QtWidgets.QMessageBox.StandardButton.Yes:
            try:
                path.unlink()
                self.refresh()
            except OSError as exc:
                self.window.error(exc)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("RomRaider Log Analyzer")
    app.setOrganizationName("SG9 Log Analysis")
    palette = app.palette()
    pg.setConfigOptions(antialias=True, background=palette.color(QtGui.QPalette.ColorRole.Base),
                        foreground=palette.color(QtGui.QPalette.ColorRole.Text))
    app.setStyleSheet("QToolBar { spacing: 9px; padding: 8px; } QPushButton { padding: 5px 9px; } "
                       "QFrame#chartPanel { border: 1px solid palette(mid); border-radius: 7px; } "
                       "QPushButton:checked { background: #397bd4; color: white; }")
    window = MainWindow()
    window.show()
    paths = [p for p in sys.argv[1:] if p.lower().endswith(".csv")]
    if paths:
        window.load_paths(paths)
    return app.exec()
