#!/usr/bin/env python3
"""
CSV Plotter — an Igor/Origin-style interactive multi-curve plotting tool.

Load any number of CSV files, pick which columns are X/Y for each, assign
a manual "parameter" value per curve (e.g. laser power, temperature) and a
label, then plot them together as:
  - Overlay (all curves on one axis, optionally colored by parameter)
  - Waterfall (curves offset vertically, in parameter or load order)
  - Power-dependence (curves colored by a colormap keyed to the parameter,
    with optional log-log axes)

Requirements:
    pip install PySide6 matplotlib pandas numpy

Run:
    python csv_plotter.py
"""

import sys
import os
import re
import csv as csv_module
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QLabel, QComboBox,
    QDoubleSpinBox, QCheckBox, QFileDialog, QHeaderView, QGroupBox,
    QFormLayout, QSplitter, QMessageBox, QColorDialog, QAbstractItemView,
    QLineEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.cm as cm
import matplotlib.colors as mcolors

try:
    from load_spe import load_spe
    SPE_AVAILABLE = True
except ImportError:
    SPE_AVAILABLE = False


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------

class Curve:
    """Holds one loaded CSV's data plus display/plot metadata."""

    _color_cycle = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    _next_color_idx = 0

    def __init__(self, path: str, df: pd.DataFrame):
        self.path = path
        self.df = df
        self.label = Path(path).stem
        self.x_col = df.columns[0]
        self.y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        self.param = 0.0          # manual parameter (power, temp, etc.)
        self.visible = True
        self.color = Curve._color_cycle[Curve._next_color_idx % len(Curve._color_cycle)]
        Curve._next_color_idx += 1

    def xy(self):
        x = pd.to_numeric(self.df[self.x_col], errors="coerce").to_numpy()
        y = pd.to_numeric(self.df[self.y_col], errors="coerce").to_numpy()
        mask = ~(np.isnan(x) | np.isnan(y))
        return x[mask], y[mask]


# ----------------------------------------------------------------------
# Matplotlib canvas
# ----------------------------------------------------------------------

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(7, 5), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def redraw(self, curves, mode, offset, logx, logy, normalize,
               colormap_name, show_legend, xlabel, ylabel, title,
               legend_loc="best", legend_fontsize=8, legend_ncol=1,
               legend_frame=True, legend_title=""):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        visible_curves = [c for c in curves if c.visible]

        use_cmap = colormap_name != "None"
        cmap = matplotlib.colormaps[colormap_name] if use_cmap else None
        if use_cmap and visible_curves:
            params = [c.param for c in visible_curves]
            pmin, pmax = min(params), max(params)
            if pmin == pmax:
                pmin, pmax = pmin - 1, pmax + 1
            norm = mcolors.Normalize(vmin=pmin, vmax=pmax)
        else:
            norm = None

        # Waterfall: order by parameter value so offsets stack sensibly
        plot_curves = visible_curves
        if mode == "waterfall":
            plot_curves = sorted(visible_curves, key=lambda c: c.param)

        for i, c in enumerate(plot_curves):
            x, y = c.xy()
            if x.size == 0:
                continue
            if normalize and np.nanmax(np.abs(y)) != 0:
                y = y / np.nanmax(np.abs(y))

            y_plot = y + (i * offset if mode == "waterfall" else 0.0)

            color = cmap(norm(c.param)) if (cmap and norm) else c.color
            self.ax.plot(x, y_plot, label=c.label, color=color, linewidth=1.4)

        if logx:
            self.ax.set_xscale("log")
        if logy:
            self.ax.set_yscale("log")

        self.ax.set_xlabel(xlabel or "X")
        self.ax.set_ylabel(ylabel or "Y")
        if title:
            self.ax.set_title(title)

        if show_legend and plot_curves:
            outside = legend_loc == "outside right"
            leg = self.ax.legend(
                fontsize=legend_fontsize,
                loc="upper left" if outside else legend_loc,
                bbox_to_anchor=(1.02, 1.0) if outside else None,
                ncol=legend_ncol,
                frameon=legend_frame,
                title=legend_title or None,
            )
            if leg and leg.get_title():
                leg.get_title().set_fontsize(legend_fontsize)

        if cmap and norm and visible_curves:
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            self.fig.colorbar(sm, ax=self.ax, label="Parameter")

        self.draw()


# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------

COLUMNS = ["File", "Label", "X col", "Y col", "Param", "Visible", "Color"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Plotter — multi-curve viewer")
        self.resize(1300, 800)

        self.curves: list[Curve] = []

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # ---------------- left: file table + controls ----------------
        left = QWidget()
        left_layout = QVBoxLayout(left)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load CSV(s)…")
        load_btn.clicked.connect(self.load_csvs)
        load_spe_btn = QPushButton("Load SPE(s)…")
        load_spe_btn.clicked.connect(self.load_spes)
        if not SPE_AVAILABLE:
            load_spe_btn.setToolTip(
                "load_spe.py was not found next to this script. "
                "Place load_spe.py in the same folder to enable .spe loading."
            )
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self.remove_selected)
        export_csv_btn = QPushButton("Export selected as CSV…")
        export_csv_btn.clicked.connect(self.export_selected_as_csv)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(load_spe_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(export_csv_btn)
        left_layout.addLayout(btn_row)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemChanged.connect(self.on_table_item_changed)
        left_layout.addWidget(self.table)

        # ---- plot options ----
        opts = QGroupBox("Plot options")
        form = QFormLayout(opts)

        self.mode_box = QComboBox()
        self.mode_box.addItems(["Overlay", "Waterfall", "Power dependence"])
        self.mode_box.currentIndexChanged.connect(self.update_plot)
        form.addRow("Mode:", self.mode_box)

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-1e9, 1e9)
        self.offset_spin.setDecimals(6)
        self.offset_spin.setValue(1.0)
        self.offset_spin.valueChanged.connect(self.update_plot)
        form.addRow("Waterfall offset:", self.offset_spin)

        self.cmap_box = QComboBox()
        self.cmap_box.addItems(["None", "viridis", "plasma", "inferno",
                                 "magma", "cividis", "turbo", "coolwarm"])
        self.cmap_box.currentIndexChanged.connect(self.update_plot)
        form.addRow("Colormap (by Param):", self.cmap_box)

        self.logx_chk = QCheckBox("Log X")
        self.logx_chk.stateChanged.connect(self.update_plot)
        self.logy_chk = QCheckBox("Log Y")
        self.logy_chk.stateChanged.connect(self.update_plot)
        log_row = QHBoxLayout()
        log_row.addWidget(self.logx_chk)
        log_row.addWidget(self.logy_chk)
        form.addRow("Axes:", log_row)

        self.norm_chk = QCheckBox("Normalize each curve to max")
        self.norm_chk.stateChanged.connect(self.update_plot)
        form.addRow("", self.norm_chk)

        self.legend_chk = QCheckBox("Show legend")
        self.legend_chk.setChecked(True)
        self.legend_chk.stateChanged.connect(self.update_plot)
        form.addRow("", self.legend_chk)

        self.legend_loc_box = QComboBox()
        self.legend_loc_box.addItems([
            "best", "upper right", "upper left", "lower left", "lower right",
            "center left", "center right", "lower center", "upper center",
            "center", "outside right",
        ])
        self.legend_loc_box.currentIndexChanged.connect(self.update_plot)
        form.addRow("Legend position:", self.legend_loc_box)

        self.legend_fontsize_spin = QDoubleSpinBox()
        self.legend_fontsize_spin.setRange(4, 32)
        self.legend_fontsize_spin.setValue(8)
        self.legend_fontsize_spin.valueChanged.connect(self.update_plot)
        form.addRow("Legend font size:", self.legend_fontsize_spin)

        self.legend_ncol_spin = QDoubleSpinBox()
        self.legend_ncol_spin.setDecimals(0)
        self.legend_ncol_spin.setRange(1, 10)
        self.legend_ncol_spin.setValue(1)
        self.legend_ncol_spin.valueChanged.connect(self.update_plot)
        form.addRow("Legend columns:", self.legend_ncol_spin)

        self.legend_frame_chk = QCheckBox("Draw frame")
        self.legend_frame_chk.setChecked(True)
        self.legend_frame_chk.stateChanged.connect(self.update_plot)
        form.addRow("", self.legend_frame_chk)

        self.legend_title_edit = QLineEdit()
        self.legend_title_edit.textChanged.connect(self.update_plot)
        form.addRow("Legend title:", self.legend_title_edit)

        self.xlabel_edit = QLineEdit()
        self.xlabel_edit.textChanged.connect(self.update_plot)
        form.addRow("X label:", self.xlabel_edit)

        self.ylabel_edit = QLineEdit()
        self.ylabel_edit.textChanged.connect(self.update_plot)
        form.addRow("Y label:", self.ylabel_edit)

        self.title_edit = QLineEdit()
        self.title_edit.textChanged.connect(self.update_plot)
        form.addRow("Title:", self.title_edit)

        left_layout.addWidget(opts)

        export_btn = QPushButton("Export figure…")
        export_btn.clicked.connect(self.export_figure)
        left_layout.addWidget(export_btn)

        splitter.addWidget(left)

        # ---------------- right: plot ----------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.canvas = PlotCanvas()
        toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(toolbar)
        right_layout.addWidget(self.canvas)
        splitter.addWidget(right)

        splitter.setSizes([480, 820])

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def load_csvs(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select CSV files", "", "CSV files (*.csv);;All files (*)"
        )
        if not paths:
            return
        for p in paths:
            try:
                df = self._read_csv(p)
            except Exception as e:
                QMessageBox.warning(self, "Load error", f"Could not read {p}:\n{e}")
                continue
            if df.shape[1] < 1:
                QMessageBox.warning(self, "Load error", f"{p} has no columns.")
                continue
            self.curves.append(Curve(p, df))
        self.refresh_table()
        self.update_plot()

    @staticmethod
    def _read_csv(path):
        # Sniff delimiter; fall back to comma. Handles tab/semicolon-delimited
        # exports too (common from spectrometer software).
        with open(path, "r", newline="", errors="ignore") as f:
            sample = f.read(4096)
        try:
            dialect = csv_module.Sniffer().sniff(sample, delimiters=",;\t")
            sep = dialect.delimiter
        except csv_module.Error:
            sep = ","
        df = pd.read_csv(path, sep=sep, engine="python")
        # If header row wasn't numeric-friendly and got parsed as data, pandas
        # already handles that via header inference; nothing more to do here.
        return df

    def load_spes(self):
        if not SPE_AVAILABLE:
            QMessageBox.warning(
                self, "SPE support unavailable",
                "load_spe.py was not found next to this script.\n"
                "Place load_spe.py in the same folder and restart the app."
            )
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select SPE files", "", "SPE files (*.spe);;All files (*)"
        )
        if not paths:
            return
        for p in paths:
            try:
                new_curves = self._curves_from_spe(p)
            except Exception as e:
                QMessageBox.warning(self, "Load error", f"Could not read {p}:\n{e}")
                continue
            self.curves.extend(new_curves)
        self.refresh_table()
        self.update_plot()

    @staticmethod
    def _curves_from_spe(path):
        """Load a .spe file and return a list of Curve objects.

        load_spe() can return several shapes depending on the file:
          - 2D array           -> single ROI, single frame  -> one curve
          - 3D array           -> single ROI, multiple frames -> one curve per frame
          - dict {'ROI': [...]}    -> multiple ROIs, single frame -> one curve per ROI
          - list of dicts           -> multiple ROIs, multiple frames -> one curve per (frame, ROI)
        Each curve is backed by a small DataFrame (Wavelength, Intensity) so
        it plugs into the same Curve/table machinery as CSV-loaded data.
        """
        data, wavelengths, _params = load_spe(path)
        stem = Path(path).stem
        curves = []

        def make_curve(y, label, wl=None):
            y = np.asarray(y).reshape(-1)
            if wl is None or len(np.asarray(wl)) != len(y):
                wl = np.arange(1, len(y) + 1)
            df = pd.DataFrame({
                "Wavelength (nm)": np.asarray(wl).reshape(-1),
                "Intensity": y,
            })
            c = Curve(path, df)
            c.label = label
            return c

        if isinstance(data, np.ndarray) and data.ndim == 2:
            curves.append(make_curve(data[:, 0], stem, wavelengths))

        elif isinstance(data, np.ndarray) and data.ndim == 3:
            n_frames = data.shape[2]
            for f in range(n_frames):
                curves.append(make_curve(data[:, 0, f], f"{stem}_frame{f}", wavelengths))

        elif isinstance(data, dict):  # multi-ROI, single frame
            wl_list = wavelengths if isinstance(wavelengths, list) else [wavelengths] * len(data["ROI"])
            for i, arr in enumerate(data["ROI"]):
                wl = wl_list[i] if i < len(wl_list) else None
                curves.append(make_curve(arr[:, 0], f"{stem}_ROI{i}", wl))

        elif isinstance(data, list):  # multi-ROI, multiple frames
            wl_list = wavelengths if isinstance(wavelengths, list) else None
            for f, frame in enumerate(data):
                for i, arr in enumerate(frame["ROI"]):
                    wl = wl_list[i] if wl_list and i < len(wl_list) else None
                    curves.append(make_curve(arr[:, 0], f"{stem}_frame{f}_ROI{i}", wl))

        else:
            raise ValueError(f"Unrecognized data structure returned by load_spe: {type(data)}")

        return curves

    def remove_selected(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self.curves):
                del self.curves[r]
        self.refresh_table()
        self.update_plot()

    def export_selected_as_csv(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        selected = [self.curves[r] for r in rows if 0 <= r < len(self.curves)]
        if not selected:
            QMessageBox.information(
                self, "Nothing selected",
                "Select one or more rows in the table first."
            )
            return

        if len(selected) == 1:
            c = selected[0]
            default_name = self._safe_filename(c.label) + ".csv"
            path, _ = QFileDialog.getSaveFileName(
                self, "Export CSV", default_name, "CSV files (*.csv)"
            )
            if not path:
                return
            self._write_curve_csv(c, path)
        else:
            directory = QFileDialog.getExistingDirectory(
                self, "Choose folder to export CSVs into"
            )
            if not directory:
                return
            for c in selected:
                filename = self._safe_filename(c.label) + ".csv"
                self._write_curve_csv(c, os.path.join(directory, filename))
            QMessageBox.information(
                self, "Export complete",
                f"Exported {len(selected)} CSV file(s) to:\n{directory}"
            )

    @staticmethod
    def _safe_filename(name):
        return re.sub(r'[^\w\-. ]', "_", name).strip() or "curve"

    @staticmethod
    def _write_curve_csv(curve, path):
        x, y = curve.xy()
        out = pd.DataFrame({str(curve.x_col): x, str(curve.y_col): y})
        out.to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Table <-> data sync
    # ------------------------------------------------------------------

    def refresh_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.curves))
        for row, c in enumerate(self.curves):
            self.table.setItem(row, 0, self._readonly_item(Path(c.path).name))

            label_item = QTableWidgetItem(c.label)
            self.table.setItem(row, 1, label_item)

            x_combo = QComboBox()
            x_combo.addItems([str(col) for col in c.df.columns])
            x_combo.setCurrentText(str(c.x_col))
            x_combo.currentTextChanged.connect(
                lambda text, curve=c: self._set_col(curve, "x_col", text))
            self.table.setCellWidget(row, 2, x_combo)

            y_combo = QComboBox()
            y_combo.addItems([str(col) for col in c.df.columns])
            y_combo.setCurrentText(str(c.y_col))
            y_combo.currentTextChanged.connect(
                lambda text, curve=c: self._set_col(curve, "y_col", text))
            self.table.setCellWidget(row, 3, y_combo)

            param_item = QTableWidgetItem(str(c.param))
            self.table.setItem(row, 4, param_item)

            vis_item = QTableWidgetItem()
            vis_item.setFlags(vis_item.flags() | Qt.ItemIsUserCheckable)
            vis_item.setCheckState(Qt.Checked if c.visible else Qt.Unchecked)
            self.table.setItem(row, 5, vis_item)

            color_btn = QPushButton()
            color_btn.setStyleSheet(f"background-color: {c.color};")
            color_btn.clicked.connect(lambda _, curve=c, btn=color_btn: self._pick_color(curve, btn))
            self.table.setCellWidget(row, 6, color_btn)

        self.table.blockSignals(False)

    @staticmethod
    def _readonly_item(text):
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _set_col(self, curve, attr, text):
        setattr(curve, attr, text)
        self.update_plot()

    def _pick_color(self, curve, btn):
        color = QColorDialog.getColor(QColor(curve.color), self)
        if color.isValid():
            curve.color = color.name()
            btn.setStyleSheet(f"background-color: {curve.color};")
            self.update_plot()

    def on_table_item_changed(self, item):
        row, col = item.row(), item.column()
        if row >= len(self.curves):
            return
        c = self.curves[row]
        if col == 1:
            c.label = item.text()
        elif col == 4:
            try:
                c.param = float(item.text())
            except ValueError:
                item.setText(str(c.param))
                return
        elif col == 5:
            c.visible = item.checkState() == Qt.Checked
        self.update_plot()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def update_plot(self):
        mode_map = {0: "overlay", 1: "waterfall", 2: "power"}
        mode = mode_map[self.mode_box.currentIndex()]
        self.canvas.redraw(
            curves=self.curves,
            mode=mode,
            offset=self.offset_spin.value(),
            logx=self.logx_chk.isChecked(),
            logy=self.logy_chk.isChecked(),
            normalize=self.norm_chk.isChecked(),
            colormap_name=self.cmap_box.currentText(),
            show_legend=self.legend_chk.isChecked(),
            xlabel=self.xlabel_edit.text(),
            ylabel=self.ylabel_edit.text(),
            title=self.title_edit.text(),
            legend_loc=self.legend_loc_box.currentText(),
            legend_fontsize=self.legend_fontsize_spin.value(),
            legend_ncol=int(self.legend_ncol_spin.value()),
            legend_frame=self.legend_frame_chk.isChecked(),
            legend_title=self.legend_title_edit.text(),
        )

    def export_figure(self):
        if not self.curves:
            QMessageBox.information(self, "Nothing to export", "Load some data first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", "plot.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if path:
            self.canvas.fig.savefig(path, dpi=300)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()