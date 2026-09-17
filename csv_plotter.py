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

import scipy
import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QLabel, QComboBox,
    QDoubleSpinBox, QCheckBox, QFileDialog, QHeaderView, QGroupBox,
    QFormLayout, QSplitter, QMessageBox, QColorDialog, QAbstractItemView,
    QLineEdit, QDialog, QTabWidget, QScrollArea
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

try:
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    from scipy.integrate import simpson
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# NumPy 2.0 renamed trapz -> trapezoid; support both versions.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ----------------------------------------------------------------------
# Small UI helpers (reduce repeated widget boilerplate)
# ----------------------------------------------------------------------

APP_STYLESHEET = """
QWidget {
    font-size: 13px;
}
QMainWindow, QDialog {
    background-color: #f5f5f7;
}
QGroupBox {
    font-weight: 600;
    border: 1px solid #d7d7db;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 12px;
    background-color: #fbfbfc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #333;
}
QPushButton {
    background-color: #ffffff;
    border: 1px solid #c7c7cc;
    border-radius: 6px;
    padding: 6px 14px;
    min-height: 18px;
}
QPushButton:hover {
    background-color: #eef2ff;
    border-color: #a9b6f5;
}
QPushButton:pressed {
    background-color: #dde4fb;
}
QPushButton:disabled {
    color: #9a9a9e;
    background-color: #f0f0f2;
}
QLineEdit, QDoubleSpinBox {
    border: 1px solid #c7c7cc;
    border-radius: 6px;
    padding: 4px 6px;
    background-color: #ffffff;
    min-height: 18px;
}
QLineEdit:focus, QDoubleSpinBox:focus {
    border: 1px solid #6a8cf5;
}
QTableWidget {
    border: 1px solid #d7d7db;
    border-radius: 6px;
    background-color: #ffffff;
    gridline-color: #e6e6ea;
    alternate-background-color: #f7f7f9;
}
QHeaderView::section {
    background-color: #eeeef1;
    border: none;
    border-bottom: 1px solid #d7d7db;
    padding: 6px;
    font-weight: 600;
}
QTabWidget::pane {
    border: 1px solid #d7d7db;
    border-radius: 8px;
    top: -1px;
    background-color: #fbfbfc;
}
QTabBar::tab {
    background: #eeeef1;
    border: 1px solid #d7d7db;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 6px 14px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #fbfbfc;
    font-weight: 600;
}
QLabel {
    color: #222;
}
"""
# Note: QComboBox is deliberately left unstyled. Any QSS applied to it makes
# Qt switch its popup from macOS's native rendering (which auto-sizes and
# colors correctly) to Qt's own generic list popup, which has proven
# unreliable here (truncated items, and on some Qt/macOS combinations,
# invisible/mismatched text color in the popup) across several attempted
# fixes. Native rendering has no such issues, so combo boxes are exempted.


class ComboBox(QComboBox):
    """Plain QComboBox alias.

    Kept as a distinct class (rather than switching every call site back to
    QComboBox) in case a future, better-tested styling approach is worth
    revisiting — but for now it deliberately adds no behavior, so every
    combo box in the app uses macOS's native popup as-is.
    """
    pass


def make_spin(min_=-1e12, max_=1e12, decimals=4, value=0.0):
    """A QDoubleSpinBox with the wide-range settings used throughout the app."""
    spin = QDoubleSpinBox()
    spin.setRange(min_, max_)
    spin.setDecimals(decimals)
    spin.setValue(value)
    return spin


def add_apply_row(layout, on_click, text="Apply"):
    """Right-aligned single button row, used by the Curve Tools panels."""
    row = QHBoxLayout()
    btn = QPushButton(text)
    btn.clicked.connect(on_click)
    row.addStretch()
    row.addWidget(btn)
    layout.addLayout(row)
    return btn


def selected_table_rows(table, reverse=False):
    return sorted({idx.row() for idx in table.selectedIndexes()}, reverse=reverse)


def make_figure_canvas(figsize):
    """A Figure + single Axes + its FigureCanvasQTAgg, matplotlib's usual trio."""
    fig = Figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(111)
    return fig, ax, FigureCanvas(fig)


def make_export_group(on_export, default_w=7.0, default_h=5.0):
    """Compact 'Export figure' box: W/H/DPI spinboxes on one line, the
    export button below — split across two rows (rather than one long row)
    so it stays narrow enough to fit a narrow panel without needing extra
    width. Returns (group_box, width_spin, height_spin, dpi_spin)."""
    group = QGroupBox("Export figure")
    layout = QVBoxLayout(group)

    spin_row = QHBoxLayout()
    width_spin = make_spin(min_=1, max_=40, decimals=2, value=default_w)
    width_spin.setMaximumWidth(55)
    height_spin = make_spin(min_=1, max_=40, decimals=2, value=default_h)
    height_spin.setMaximumWidth(55)
    dpi_spin = make_spin(min_=50, max_=1200, decimals=0, value=300)
    dpi_spin.setMaximumWidth(55)

    spin_row.addWidget(QLabel("W:"))
    spin_row.addWidget(width_spin)
    spin_row.addWidget(QLabel("H:"))
    spin_row.addWidget(height_spin)
    spin_row.addWidget(QLabel("DPI:"))
    spin_row.addWidget(dpi_spin)
    spin_row.addStretch()
    layout.addLayout(spin_row)

    export_btn = QPushButton("Export figure…")
    export_btn.clicked.connect(on_export)
    layout.addWidget(export_btn)

    return group, width_spin, height_spin, dpi_spin


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

# ----------------------------------------------------------------------
# Data point editor (delete individual points, Igor/Origin table style)
# ----------------------------------------------------------------------

class EditDataPanel(QWidget):
    """Delete individual data points from one curve, Igor/Origin table style."""

    def __init__(self, curve: Curve, on_change, parent=None):
        super().__init__(parent)
        self.curve = curve
        self.on_change = on_change
        self.working_df = curve.df.reset_index(drop=True)

        layout = QVBoxLayout(self)

        info = QLabel(
            "Select row(s) and click \u201cDelete selected\u201d (or press "
            "Delete/Backspace) to remove data points, then \u201cApply "
            "changes\u201d to update the plot."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self._populate()
        layout.addWidget(self.table)

        delete_row = QHBoxLayout()
        delete_btn = QPushButton("Delete selected row(s)")
        delete_btn.clicked.connect(self.delete_selected)
        delete_row.addWidget(delete_btn)
        delete_row.addStretch()
        self.count_label = QLabel()
        self._update_count_label()
        delete_row.addWidget(self.count_label)
        layout.addLayout(delete_row)

        apply_row = QHBoxLayout()
        apply_btn = QPushButton("Apply changes")
        apply_btn.clicked.connect(self.apply_changes)
        apply_row.addStretch()
        apply_row.addWidget(apply_btn)
        layout.addLayout(apply_row)

    def _populate(self):
        self.table.setUpdatesEnabled(False)
        self.table.blockSignals(True)
        cols = list(self.working_df.columns)
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels([str(c) for c in cols])
        self.table.setRowCount(len(self.working_df))
        for row in range(len(self.working_df)):
            for col in range(len(cols)):
                val = self.working_df.iat[row, col]
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.blockSignals(False)
        self.table.setUpdatesEnabled(True)

    def _update_count_label(self):
        self.count_label.setText(f"{len(self.working_df)} point(s)")

    def delete_selected(self):
        rows = selected_table_rows(self.table, reverse=True)
        if not rows:
            return
        self.working_df = self.working_df.drop(self.working_df.index[rows]).reset_index(drop=True)
        self._populate()
        self._update_count_label()

    def apply_changes(self):
        self.curve.df = self.working_df
        self.on_change()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(event)


COLUMNS = ["File", "Label", "X col", "Y col", "Param", "Visible", "Color"]


class BackgroundSubtractPanel(QWidget):
    MODE_CURVE = "Subtract another curve (pointwise)"
    MODE_RANGE = "Subtract constant: mean over X range"
    MODE_CONSTANT = "Subtract constant: manual value"

    def __init__(self, target_curves, all_curves, on_change, parent=None):
        super().__init__(parent)
        self.target_curves = target_curves
        self.on_change = on_change

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Applying to {len(target_curves)} selected curve(s)."))

        form = QFormLayout()

        self.mode_box = ComboBox()
        self.mode_box.addItems([self.MODE_CURVE, self.MODE_RANGE, self.MODE_CONSTANT])
        self.mode_box.currentTextChanged.connect(self._on_mode_changed)
        form.addRow("Method:", self.mode_box)

        self.bg_curve_box = ComboBox()
        eligible = [c for c in all_curves if c not in target_curves]
        for c in eligible:
            self.bg_curve_box.addItem(c.label, c)
        form.addRow("Background curve:", self.bg_curve_box)

        xs = [c.xy()[0] for c in target_curves if c.xy()[0].size]
        if xs:
            allx = np.concatenate(xs)
            xmin_default, xmax_default = float(np.nanmin(allx)), float(np.nanmax(allx))
        else:
            xmin_default, xmax_default = 0.0, 1.0
        span = xmax_default - xmin_default

        self.xmin_spin = make_spin(decimals=4, value=xmin_default)
        form.addRow("X range min:", self.xmin_spin)

        self.xmax_spin = make_spin(decimals=4, value=xmin_default + 0.05 * span if span > 0 else xmax_default)
        form.addRow("X range max:", self.xmax_spin)

        self.constant_spin = make_spin(decimals=6, value=0.0)
        form.addRow("Constant value:", self.constant_spin)

        layout.addLayout(form)
        layout.addStretch()
        add_apply_row(layout, self.apply_changes)

        self._on_mode_changed(self.mode_box.currentText())

    def _on_mode_changed(self, mode):
        self.bg_curve_box.setEnabled(mode == self.MODE_CURVE)
        is_range = mode == self.MODE_RANGE
        self.xmin_spin.setEnabled(is_range)
        self.xmax_spin.setEnabled(is_range)
        self.constant_spin.setEnabled(mode == self.MODE_CONSTANT)

    def apply_changes(self):
        if self.mode_box.currentText() == self.MODE_CURVE and self.bg_curve_box.count() == 0:
            QMessageBox.warning(
                self, "No background curve available",
                "There is no other loaded curve to use as a background reference."
            )
            return
        for c in self.target_curves:
            self._apply_to(c)
        self.on_change()

    def _apply_to(self, curve):
        mode = self.mode_box.currentText()
        full_x = pd.to_numeric(curve.df[curve.x_col], errors="coerce").to_numpy()
        full_y = pd.to_numeric(curve.df[curve.y_col], errors="coerce").to_numpy()

        if mode == self.MODE_CURVE:
            bg_curve = self.bg_curve_box.currentData()
            if bg_curve is None:
                return
            bg_x, bg_y = bg_curve.xy()
            order = np.argsort(bg_x)
            bg_interp = np.interp(full_x, bg_x[order], bg_y[order])
            new_y = full_y - bg_interp
        elif mode == self.MODE_RANGE:
            xmin, xmax = self.xmin_spin.value(), self.xmax_spin.value()
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            mask = (full_x >= xmin) & (full_x <= xmax)
            bg_value = np.nanmean(full_y[mask]) if mask.any() else 0.0
            new_y = full_y - bg_value
        else:  # MODE_CONSTANT
            new_y = full_y - self.constant_spin.value()

        curve.df = curve.df.copy()
        curve.df[curve.y_col] = new_y


class ScalePanel(QWidget):
    def __init__(self, target_curves, on_change, parent=None):
        super().__init__(parent)
        self.target_curves = target_curves
        self.on_change = on_change

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Applying to {len(target_curves)} selected curve(s)."))

        form = QFormLayout()

        self.op_box = ComboBox()
        self.op_box.addItems(["Divide by", "Multiply by"])
        form.addRow("Operation:", self.op_box)

        self.value_spin = make_spin(decimals=6, value=1.0)
        form.addRow("Value:", self.value_spin)

        layout.addLayout(form)
        layout.addStretch()
        add_apply_row(layout, self.apply_changes)

    def apply_changes(self):
        if self.op_box.currentText() == "Divide by" and self.value_spin.value() == 0:
            QMessageBox.warning(self, "Invalid value", "Cannot divide by zero.")
            return
        value = self.value_spin.value()
        for curve in self.target_curves:
            full_y = pd.to_numeric(curve.df[curve.y_col], errors="coerce").to_numpy()
            new_y = full_y / value if self.op_box.currentText() == "Divide by" else full_y * value
            curve.df = curve.df.copy()
            curve.df[curve.y_col] = new_y
        self.on_change()


class IntegratePanel(QWidget):
    """Integrate (area under curve) over an X range, one or more curves at once."""

    def __init__(self, target_curves, on_change, parent=None):
        super().__init__(parent)
        self.target_curves = target_curves
        self.on_change = on_change
        self.last_results = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Applying to {len(target_curves)} selected curve(s)."))

        form = QFormLayout()

        xs = [c.xy()[0] for c in target_curves if c.xy()[0].size]
        if xs:
            allx = np.concatenate(xs)
            xmin_default, xmax_default = float(np.nanmin(allx)), float(np.nanmax(allx))
        else:
            xmin_default, xmax_default = 0.0, 1.0

        self.xmin_spin = make_spin(decimals=4, value=xmin_default)
        self.xmin_spin.valueChanged.connect(self._update_cursors)
        form.addRow("X range min:", self.xmin_spin)

        self.xmax_spin = make_spin(decimals=4, value=xmax_default)
        self.xmax_spin.valueChanged.connect(self._update_cursors)
        form.addRow("X range max:", self.xmax_spin)

        self.method_box = ComboBox()
        methods = ["Trapezoidal"]
        if SCIPY_AVAILABLE:
            methods.append("Simpson's rule")
        self.method_box.addItems(methods)
        form.addRow("Method:", self.method_box)

        self.baseline_chk = QCheckBox("Subtract flat baseline (min Y in range) first")
        form.addRow("", self.baseline_chk)

        self.store_param_chk = QCheckBox("Store result in Param column")
        form.addRow("", self.store_param_chk)

        layout.addLayout(form)

        layout.addWidget(QLabel("Drag the red dashed lines to set the range, or type values above."))
        self.fig, self.ax, self.preview_canvas = make_figure_canvas((6, 3.2))
        layout.addWidget(self.preview_canvas)
        self._dragging = None
        self.min_line = self.max_line = self.span = None
        self._plot_data()
        self.preview_canvas.mpl_connect("button_press_event", self._on_press)
        self.preview_canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.preview_canvas.mpl_connect("button_release_event", self._on_release)

        self.results_label = QLabel("Click Compute to see integrated values.")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.results_label)
        layout.addStretch()

        copy_row = QHBoxLayout()
        copy_btn = QPushButton("Copy results")
        copy_btn.clicked.connect(self.copy_results)
        copy_row.addWidget(copy_btn)
        copy_row.addStretch()
        layout.addLayout(copy_row)

        add_apply_row(layout, self.compute, "Compute")

    def _plot_data(self):
        self.ax.clear()
        for c in self.target_curves:
            x, y = c.xy()
            if x.size:
                self.ax.plot(x, y, color=c.color, linewidth=1.2, label=c.label)
        if len(self.target_curves) > 1:
            self.ax.legend(fontsize=7)
        xmin, xmax = self.xmin_spin.value(), self.xmax_spin.value()
        self.min_line = self.ax.axvline(xmin, color="red", linestyle="--", linewidth=1.5)
        self.max_line = self.ax.axvline(xmax, color="red", linestyle="--", linewidth=1.5)
        self.span = self.ax.axvspan(min(xmin, xmax), max(xmin, xmax), color="red", alpha=0.15)
        self.preview_canvas.draw()

    def _update_cursors(self):
        if self.min_line is None:
            return
        xmin, xmax = self.xmin_spin.value(), self.xmax_spin.value()
        self.min_line.set_xdata([xmin, xmin])
        self.max_line.set_xdata([xmax, xmax])
        self.span.remove()
        self.span = self.ax.axvspan(min(xmin, xmax), max(xmin, xmax), color="red", alpha=0.15)
        self.preview_canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        x0, x1 = self.ax.get_xlim()
        tol = abs(x1 - x0) * 0.02
        xmin, xmax = self.xmin_spin.value(), self.xmax_spin.value()
        if abs(event.xdata - xmin) <= tol:
            self._dragging = "min"
        elif abs(event.xdata - xmax) <= tol:
            self._dragging = "max"

    def _on_motion(self, event):
        if self._dragging is None or event.inaxes != self.ax or event.xdata is None:
            return
        if self._dragging == "min":
            self.xmin_spin.setValue(event.xdata)
        else:
            self.xmax_spin.setValue(event.xdata)

    def _on_release(self, event):
        self._dragging = None

    def compute(self):
        xmin, xmax = self.xmin_spin.value(), self.xmax_spin.value()
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        method = self.method_box.currentText()

        results = []
        for c in self.target_curves:
            x, y = c.xy()
            mask = (x >= xmin) & (x <= xmax)
            xs_, ys_ = x[mask], y[mask]
            if xs_.size < 2:
                results.append((c, float("nan")))
                continue
            order = np.argsort(xs_)
            xs_, ys_ = xs_[order], ys_[order]
            if self.baseline_chk.isChecked():
                ys_ = ys_ - np.nanmin(ys_)
            if method == "Simpson's rule" and SCIPY_AVAILABLE:
                value = float(simpson(ys_, x=xs_))
            else:
                value = float(_trapezoid(ys_, x=xs_))
            results.append((c, value))

        self.last_results = results
        lines = [f"{c.label}: {v:.6g}" for c, v in results] if results else ["No curves to integrate."]
        self.results_label.setText("\n".join(lines))

        if self.store_param_chk.isChecked() and results:
            for c, v in results:
                if not np.isnan(v):
                    c.param = v
            self.on_change()

    def copy_results(self):
        text = self.results_label.text()
        if not text or text.startswith("Click Compute"):
            QMessageBox.information(self, "No results yet", "Click Compute first.")
            return
        QApplication.clipboard().setText(text)


# ----------------------------------------------------------------------
# Gaussian peak fitting (sum of Gaussians + optional baseline)
# ----------------------------------------------------------------------

def _baseline_values(x, base_params, mode):
    if mode == "constant":
        return np.full_like(x, base_params[0], dtype=float)
    elif mode == "linear":
        return base_params[0] + base_params[1] * x
    return np.zeros_like(x, dtype=float)


def _build_gaussian_model(n_peaks, baseline_mode):
    def model(x, *params):
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x)
        for i in range(n_peaks):
            amp, cen, sigma = params[3 * i:3 * i + 3]
            y += amp * np.exp(-((x - cen) ** 2) / (2 * sigma ** 2))
        y += _baseline_values(x, params[3 * n_peaks:], baseline_mode)
        return y
    return model


def _estimate_initial_params(x, y, n_peaks, baseline_mode):
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))

    if baseline_mode == "linear":
        c0 = float(y[0])
        c1 = float((y[-1] - y[0]) / (x[-1] - x[0])) if x[-1] != x[0] else 0.0
        baseline_vals = c0 + c1 * x
        base_p0 = [c0, c1]
    elif baseline_mode == "constant":
        c0 = float(np.nanmin(y))
        baseline_vals = np.full_like(y, c0)
        base_p0 = [c0]
    else:
        baseline_vals = np.zeros_like(y)
        base_p0 = []

    residual = y - baseline_vals
    max_resid = float(np.nanmax(residual)) if residual.size else 1.0
    height_thresh = 0.1 * max_resid if max_resid > 0 else None
    min_distance = max(1, x.size // (4 * max(n_peaks, 1)))
    prominence_thresh = 0.15 * max_resid if max_resid > 0 else None
    peak_idx, properties = find_peaks(
        residual, height=height_thresh, distance=min_distance, prominence=prominence_thresh
    )

    if len(peak_idx) == 0:
        peak_idx = np.array([int(np.argmax(residual))])
        heights = residual[peak_idx]
    else:
        heights = properties.get("peak_heights", residual[peak_idx])

    order = np.argsort(heights)[::-1]
    peak_idx = np.asarray(peak_idx)[order][:n_peaks]
    heights = np.asarray(heights)[order][:n_peaks]

    if len(peak_idx) < n_peaks:
        extra_needed = n_peaks - len(peak_idx)
        extra_x = np.linspace(xmin, xmax, extra_needed + 2)[1:-1]
        for ex in extra_x:
            idx = int(np.argmin(np.abs(x - ex)))
            peak_idx = np.append(peak_idx, idx)
            heights = np.append(heights, residual[idx])

    span = xmax - xmin if xmax > xmin else 1.0
    sigma0 = span / (6 * max(n_peaks, 1))

    p0 = []
    for idx, h in zip(peak_idx, heights):
        amp0 = float(h) if h > 0 else (max_resid if max_resid > 0 else 1.0)
        p0 += [amp0, float(x[idx]), sigma0]
    p0 += base_p0
    return p0


class GaussianFitPanel(QWidget):
    def __init__(self, curve: Curve, add_curves_callback, parent=None):
        super().__init__(parent)
        self.curve = curve
        self.add_curves_callback = add_curves_callback
        self.fit_result = None

        self.full_x, self.full_y = curve.xy()

        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.xmin_spin = make_spin(decimals=4)
        self.xmax_spin = make_spin(decimals=4)
        if self.full_x.size:
            self.xmin_spin.setValue(float(np.nanmin(self.full_x)))
            self.xmax_spin.setValue(float(np.nanmax(self.full_x)))
        form.addRow("Fit X min:", self.xmin_spin)
        form.addRow("Fit X max:", self.xmax_spin)

        self.n_peaks_spin = make_spin(min_=1, max_=6, decimals=0, value=1)
        form.addRow("Number of peaks:", self.n_peaks_spin)

        self.baseline_box = ComboBox()
        self.baseline_box.addItems(["none", "constant", "linear"])
        self.baseline_box.setCurrentText("constant")
        form.addRow("Baseline:", self.baseline_box)

        layout.addLayout(form)

        run_btn = QPushButton("Run fit")
        run_btn.clicked.connect(self.run_fit)
        layout.addWidget(run_btn)

        self.fig, self.ax, self.preview_canvas = make_figure_canvas((5, 3.2))
        layout.addWidget(self.preview_canvas)
        self._plot_data_only()

        self.results_label = QLabel("Run a fit to see peak parameters.")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.results_label)

        opts_row = QHBoxLayout()
        self.add_total_chk = QCheckBox("Add total fit curve")
        self.add_total_chk.setChecked(True)
        self.add_peaks_chk = QCheckBox("Add individual peak curves")
        copy_btn = QPushButton("Copy results")
        copy_btn.clicked.connect(self.copy_results)
        opts_row.addWidget(self.add_total_chk)
        opts_row.addWidget(self.add_peaks_chk)
        opts_row.addStretch()
        opts_row.addWidget(copy_btn)
        layout.addLayout(opts_row)

        self.apply_btn = add_apply_row(layout, self.add_curves_to_plot, "Add Fit Curve(s) to Plot")
        self.apply_btn.setEnabled(False)

    def _plot_data_only(self):
        self.ax.clear()
        self.ax.plot(self.full_x, self.full_y, '.', color='#1f77b4', markersize=3, label='Data')
        self.ax.legend(fontsize=8)
        self.preview_canvas.draw()

    def run_fit(self):
        xmin, xmax = self.xmin_spin.value(), self.xmax_spin.value()
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        mask = (self.full_x >= xmin) & (self.full_x <= xmax)
        x, y = self.full_x[mask], self.full_y[mask]
        if x.size < 5:
            QMessageBox.warning(self, "Not enough data", "Selected X range has too few points to fit.")
            return

        n_peaks = int(self.n_peaks_spin.value())
        baseline_mode = self.baseline_box.currentText()

        model = _build_gaussian_model(n_peaks, baseline_mode)
        p0 = _estimate_initial_params(x, y, n_peaks, baseline_mode)

        span = (x.max() - x.min()) if x.max() > x.min() else 1.0
        lower, upper = [], []
        for _ in range(n_peaks):
            lower += [-np.inf, x.min() - span, 1e-8]
            upper += [np.inf, x.max() + span, span]
        if baseline_mode == "linear":
            lower += [-np.inf, -np.inf]
            upper += [np.inf, np.inf]
        elif baseline_mode == "constant":
            lower += [-np.inf]
            upper += [np.inf]

        try:
            popt, _pcov = curve_fit(model, x, y, p0=p0, bounds=(lower, upper), maxfev=50000)
        except Exception as e:
            QMessageBox.warning(self, "Fit failed", f"curve_fit did not converge:\n{e}")
            return

        y_fit = model(x, *popt)
        ss_res = float(np.sum((y - y_fit) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        self.fit_result = {
            "model": model, "popt": popt, "n_peaks": n_peaks,
            "baseline_mode": baseline_mode, "x": x,
        }

        x_dense = np.linspace(x.min(), x.max(), 400)
        self.ax.clear()
        self.ax.plot(self.full_x, self.full_y, '.', color='#1f77b4', markersize=3, label='Data')
        self.ax.plot(x_dense, model(x_dense, *popt), '-', color='#d62728', linewidth=1.6, label='Fit')
        base = _baseline_values(x_dense, popt[3 * n_peaks:], baseline_mode)
        for i in range(n_peaks):
            amp, cen, sigma = popt[3 * i:3 * i + 3]
            comp = amp * np.exp(-((x_dense - cen) ** 2) / (2 * sigma ** 2)) + base
            self.ax.plot(x_dense, comp, '--', linewidth=1, alpha=0.7, label=f'Peak {i + 1}')
            fwhm = 2.3548 * abs(sigma)
            half_max_y = amp / 2 + _baseline_values(np.array([cen]), popt[3 * n_peaks:], baseline_mode)[0]
            self.ax.plot(
                [cen - fwhm / 2, cen + fwhm / 2], [half_max_y, half_max_y],
                '-', color='gray', linewidth=1, alpha=0.8
            )
            self.ax.annotate(
                f"FWHM={fwhm:.3g}", xy=(cen, half_max_y), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=7, color="gray"
            )
        self.ax.legend(fontsize=7)
        self.preview_canvas.draw()

        lines = [f"R\u00b2 = {r2:.5f}"]
        for i in range(n_peaks):
            amp, cen, sigma = popt[3 * i:3 * i + 3]
            fwhm = 2.3548 * abs(sigma)
            lines.append(
                f"Peak {i + 1}: center={cen:.5g}, amplitude={amp:.5g}, "
                f"sigma={abs(sigma):.5g}, FWHM={fwhm:.5g}"
            )
        base_p = popt[3 * n_peaks:]
        if baseline_mode == "constant":
            lines.append(f"Baseline: constant={base_p[0]:.5g}")
        elif baseline_mode == "linear":
            lines.append(f"Baseline: intercept={base_p[0]:.5g}, slope={base_p[1]:.5g}")
        self.results_label.setText("\n".join(lines))

        self.apply_btn.setEnabled(True)

    def copy_results(self):
        text = self.results_label.text()
        if not text or text.startswith("Run a fit"):
            QMessageBox.information(self, "No results yet", "Run a fit first.")
            return
        QApplication.clipboard().setText(text)

    def add_curves_to_plot(self):
        curves = self._build_result_curves(self.curve)
        if curves:
            self.add_curves_callback(curves)

    def _build_result_curves(self, original_curve):
        if not self.fit_result:
            return []
        model = self.fit_result["model"]
        popt = self.fit_result["popt"]
        n_peaks = self.fit_result["n_peaks"]
        baseline_mode = self.fit_result["baseline_mode"]
        x = self.fit_result["x"]
        x_dense = np.linspace(x.min(), x.max(), 400)

        curves = []
        if self.add_total_chk.isChecked():
            df = pd.DataFrame({"X": x_dense, "Y": model(x_dense, *popt)})
            c = Curve(original_curve.path, df)
            c.label = f"{original_curve.label}_fit"
            curves.append(c)
        if self.add_peaks_chk.isChecked():
            base = _baseline_values(x_dense, popt[3 * n_peaks:], baseline_mode)
            for i in range(n_peaks):
                amp, cen, sigma = popt[3 * i:3 * i + 3]
                fwhm = 2.3548 * abs(sigma)
                comp = amp * np.exp(-((x_dense - cen) ** 2) / (2 * sigma ** 2)) + base
                df = pd.DataFrame({"X": x_dense, "Y": comp})
                c = Curve(original_curve.path, df)
                c.label = f"{original_curve.label}_peak{i + 1} (center={cen:.4g}, FWHM={fwhm:.4g})"
                curves.append(c)
        return curves


class CurveToolsDialog(QDialog):
    def __init__(self, main_window, selected_curves, all_curves, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Curve Tools")
        self.resize(680, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        single = selected_curves[0] if len(selected_curves) == 1 else None

        # --- Edit Data (needs exactly one curve) ---
        if single is not None:
            tabs.addTab(EditDataPanel(single, main_window.update_plot), "Edit Data")
        else:
            tabs.addTab(self._placeholder(
                "Select exactly one curve in the table, then reopen Curve Tools."
            ), "Edit Data")

        # --- Subtract Background (needs one or more curves) ---
        if selected_curves:
            tabs.addTab(
                BackgroundSubtractPanel(selected_curves, all_curves, main_window.update_plot),
                "Subtract Background"
            )
        else:
            tabs.addTab(self._placeholder(
                "Select one or more curves in the table, then reopen Curve Tools."
            ), "Subtract Background")

        # --- Scale (needs one or more curves) ---
        if selected_curves:
            tabs.addTab(ScalePanel(selected_curves, main_window.update_plot), "Scale")
        else:
            tabs.addTab(self._placeholder(
                "Select one or more curves in the table, then reopen Curve Tools."
            ), "Scale")

        # --- Integrate (needs one or more curves) ---
        if selected_curves:
            tabs.addTab(IntegratePanel(selected_curves, main_window.update_plot), "Integrate")
        else:
            tabs.addTab(self._placeholder(
                "Select one or more curves in the table, then reopen Curve Tools."
            ), "Integrate")

        # --- Fit Gaussian (needs exactly one curve, and scipy) ---
        if not SCIPY_AVAILABLE:
            tabs.addTab(self._placeholder(
                "Gaussian fitting requires scipy. Install it with:\n\npip install scipy"
            ), "Fit Gaussian")
        elif single is not None:
            def add_curves(new_curves):
                main_window.curves.extend(new_curves)
                main_window.refresh_table()
                main_window.update_plot()
            tabs.addTab(GaussianFitPanel(single, add_curves), "Fit Gaussian")
        else:
            tabs.addTab(self._placeholder(
                "Select exactly one curve in the table, then reopen Curve Tools."
            ), "Fit Gaussian")

        close_row = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addStretch()
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

    @staticmethod
    def _placeholder(text):
        w = QWidget()
        l = QVBoxLayout(w)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        l.addStretch()
        l.addWidget(label)
        l.addStretch()
        return w


# ----------------------------------------------------------------------
# Polar plot builder (manual angle/radius data entry)
# ----------------------------------------------------------------------

class PolarSeries:
    """One manually-entered polar trace: parallel lists of raw text cells
    (kept as strings so a still-being-typed/invalid row doesn't wipe data),
    parsed to numeric theta/r only when plotting."""

    def __init__(self, label):
        self.label = label
        self.angle_strs = []
        self.radius_strs = []
        self.unit = "Degrees"   # or "Radians"
        self.visible = True
        self.color = Curve._color_cycle[Curve._next_color_idx % len(Curve._color_cycle)]
        Curve._next_color_idx += 1

    def theta_r(self):
        theta, r = [], []
        for a, rad in zip(self.angle_strs, self.radius_strs):
            try:
                a_val, r_val = float(a), float(rad)
            except (TypeError, ValueError):
                continue
            theta.append(a_val)
            r.append(r_val)
        theta = np.array(theta, dtype=float)
        if self.unit == "Degrees":
            theta = np.deg2rad(theta)
        return theta, np.array(r, dtype=float)


class PointsTable(QTableWidget):
    """QTableWidget with Delete/Backspace wired to a row-removal callback."""

    def __init__(self, on_delete_rows, parent=None):
        super().__init__(parent)
        self.on_delete_rows = on_delete_rows

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.on_delete_rows()
        else:
            super().keyPressEvent(event)


class PolarCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 6), tight_layout=True)
        self.ax = self.fig.add_subplot(111, projection="polar")
        super().__init__(self.fig)

    def redraw(self, series_list, theta_zero, theta_dir, r_max, show_grid,
               show_legend, title, style):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111, projection="polar")
        self.ax.set_theta_zero_location(theta_zero)
        self.ax.set_theta_direction(theta_dir)

        marker = "o" if style in ("Markers", "Line + Markers") else None
        linestyle = "-" if style in ("Line", "Line + Markers") else "None"

        any_plotted = False
        for s in series_list:
            if not s.visible:
                continue
            theta, r = s.theta_r()
            if theta.size == 0:
                continue
            any_plotted = True
            self.ax.plot(theta, r, color=s.color, marker=marker, linestyle=linestyle,
                         linewidth=1.4, markersize=4, label=s.label)

        if r_max and r_max > 0:
            self.ax.set_rmax(r_max)
        self.ax.grid(show_grid)
        if title:
            self.ax.set_title(title)
        if show_legend and any_plotted:
            self.ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.3, 1.1))

        self.draw()


class PolarPlotDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Polar Plot Builder")
        self.resize(1000, 650)
        self.series = [PolarSeries("Series 1")]
        self.current_index = 0

        splitter = QSplitter(Qt.Horizontal)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.addWidget(splitter)

        # ---------------- left: series + points ----------------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 8, 0)

        series_row = QHBoxLayout()
        self.series_box = ComboBox()
        self.series_box.currentIndexChanged.connect(self._on_series_switch)
        new_btn = QPushButton("New")
        new_btn.clicked.connect(self._new_series)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_series)
        series_row.addWidget(self.series_box)
        series_row.addWidget(new_btn)
        series_row.addWidget(remove_btn)
        left_layout.addLayout(series_row)

        form = QFormLayout()

        self.label_edit = QLineEdit()
        self.label_edit.textChanged.connect(self._on_label_changed)
        form.addRow("Label:", self.label_edit)

        self.unit_box = ComboBox()
        self.unit_box.addItems(["Degrees", "Radians"])
        self.unit_box.currentTextChanged.connect(self._on_unit_changed)
        form.addRow("Angle unit:", self.unit_box)

        self.visible_chk = QCheckBox("Visible")
        self.visible_chk.setChecked(True)
        self.visible_chk.toggled.connect(self._on_visible_changed)
        form.addRow("", self.visible_chk)

        self.color_btn = QPushButton()
        self.color_btn.clicked.connect(self._pick_color)
        form.addRow("Color:", self.color_btn)

        left_layout.addLayout(form)

        self.points_table = PointsTable(self._delete_selected_points)
        self.points_table.setColumnCount(2)
        self.points_table.setHorizontalHeaderLabels(["Angle", "Radius"])
        self.points_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.points_table.setAlternatingRowColors(True)
        self.points_table.itemChanged.connect(self._sync_from_table)
        left_layout.addWidget(self.points_table)

        pt_btn_row = QHBoxLayout()
        add_pt_btn = QPushButton("Add point row")
        add_pt_btn.clicked.connect(self._add_point_row)
        del_pt_btn = QPushButton("Delete selected row(s)")
        del_pt_btn.clicked.connect(self._delete_selected_points)
        pt_btn_row.addWidget(add_pt_btn)
        pt_btn_row.addWidget(del_pt_btn)
        left_layout.addLayout(pt_btn_row)

        splitter.addWidget(left)

        # ---------------- right: preview + plot options ----------------
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.canvas = PolarCanvas()
        toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(toolbar)
        right_layout.addWidget(self.canvas)

        opts = QGroupBox("Plot options")
        opts_form = QFormLayout(opts)

        self.zero_box = ComboBox()
        self.zero_box.addItems(["E", "N", "W", "S", "NE", "NW", "SE", "SW"])
        self.zero_box.currentIndexChanged.connect(self._redraw)
        opts_form.addRow("Theta zero location:", self.zero_box)

        self.dir_box = ComboBox()
        self.dir_box.addItems(["Counterclockwise", "Clockwise"])
        self.dir_box.currentIndexChanged.connect(self._redraw)
        opts_form.addRow("Theta direction:", self.dir_box)

        self.style_box = ComboBox()
        self.style_box.addItems(["Line", "Markers", "Line + Markers"])
        self.style_box.currentIndexChanged.connect(self._redraw)
        opts_form.addRow("Style:", self.style_box)

        self.rmax_spin = make_spin(min_=0, max_=1e9, decimals=4, value=0.0)
        self.rmax_spin.valueChanged.connect(self._redraw)
        opts_form.addRow("Radial max (0 = auto):", self.rmax_spin)

        self.grid_chk = QCheckBox("Show grid")
        self.grid_chk.setChecked(True)
        self.grid_chk.toggled.connect(self._redraw)
        opts_form.addRow("", self.grid_chk)

        self.legend_chk = QCheckBox("Show legend")
        self.legend_chk.setChecked(True)
        self.legend_chk.toggled.connect(self._redraw)
        opts_form.addRow("", self.legend_chk)

        self.title_edit = QLineEdit()
        self.title_edit.textChanged.connect(self._redraw)
        opts_form.addRow("Title:", self.title_edit)

        right_layout.addWidget(opts)

        export_group, self.fig_width_spin, self.fig_height_spin, self.fig_dpi_spin = \
            make_export_group(self.export_figure, default_w=7.0, default_h=7.0)
        right_layout.addWidget(export_group)

        close_row = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addStretch()
        close_row.addWidget(close_btn)
        right_layout.addLayout(close_row)

        splitter.addWidget(right)
        splitter.setSizes([380, 620])

        self._refresh_series_box()
        self._load_series_into_form(0)
        self._redraw()

    # ------------------------------------------------------------------
    # Series management
    # ------------------------------------------------------------------

    def _refresh_series_box(self):
        self.series_box.blockSignals(True)
        self.series_box.clear()
        self.series_box.addItems([s.label for s in self.series])
        self.series_box.setCurrentIndex(self.current_index)
        self.series_box.blockSignals(False)

    def _new_series(self):
        self.series.append(PolarSeries(f"Series {len(self.series) + 1}"))
        self.current_index = len(self.series) - 1
        self._refresh_series_box()
        self._load_series_into_form(self.current_index)
        self._redraw()

    def _remove_series(self):
        if len(self.series) <= 1:
            QMessageBox.information(self, "Can't remove", "At least one series is required.")
            return
        del self.series[self.current_index]
        self.current_index = max(0, self.current_index - 1)
        self._refresh_series_box()
        self._load_series_into_form(self.current_index)
        self._redraw()

    def _on_series_switch(self, index):
        if index < 0 or index == self.current_index:
            return
        self.current_index = index
        self._load_series_into_form(index)
        self._redraw()

    def _load_series_into_form(self, index):
        s = self.series[index]
        self.label_edit.blockSignals(True)
        self.label_edit.setText(s.label)
        self.label_edit.blockSignals(False)

        self.unit_box.blockSignals(True)
        self.unit_box.setCurrentText(s.unit)
        self.unit_box.blockSignals(False)

        self.visible_chk.blockSignals(True)
        self.visible_chk.setChecked(s.visible)
        self.visible_chk.blockSignals(False)

        self.color_btn.setStyleSheet(f"background-color: {s.color};")

        self.points_table.blockSignals(True)
        rows = max(len(s.angle_strs), 1)
        self.points_table.setRowCount(rows)
        for r in range(rows):
            a = s.angle_strs[r] if r < len(s.angle_strs) else ""
            rad = s.radius_strs[r] if r < len(s.radius_strs) else ""
            self.points_table.setItem(r, 0, QTableWidgetItem(a))
            self.points_table.setItem(r, 1, QTableWidgetItem(rad))
        self.points_table.blockSignals(False)

    def _current_series(self):
        return self.series[self.current_index]

    # ------------------------------------------------------------------
    # Per-series field edits
    # ------------------------------------------------------------------

    def _on_label_changed(self, text):
        s = self._current_series()
        s.label = text
        self.series_box.blockSignals(True)
        self.series_box.setItemText(self.current_index, text)
        self.series_box.blockSignals(False)
        self._redraw()

    def _on_unit_changed(self, text):
        self._current_series().unit = text
        self._redraw()

    def _on_visible_changed(self, checked):
        self._current_series().visible = checked
        self._redraw()

    def _pick_color(self):
        s = self._current_series()
        color = QColorDialog.getColor(QColor(s.color), self)
        if color.isValid():
            s.color = color.name()
            self.color_btn.setStyleSheet(f"background-color: {s.color};")
            self._redraw()

    # ------------------------------------------------------------------
    # Points table
    # ------------------------------------------------------------------

    def _add_point_row(self):
        row = self.points_table.rowCount()
        self.points_table.insertRow(row)
        self.points_table.blockSignals(True)
        self.points_table.setItem(row, 0, QTableWidgetItem(""))
        self.points_table.setItem(row, 1, QTableWidgetItem(""))
        self.points_table.blockSignals(False)
        self._sync_from_table()

    def _delete_selected_points(self):
        rows = selected_table_rows(self.points_table, reverse=True)
        if not rows:
            return
        for r in rows:
            self.points_table.removeRow(r)
        self._sync_from_table()

    def _sync_from_table(self):
        s = self._current_series()
        angle_strs, radius_strs = [], []
        for row in range(self.points_table.rowCount()):
            a_item = self.points_table.item(row, 0)
            r_item = self.points_table.item(row, 1)
            angle_strs.append(a_item.text() if a_item else "")
            radius_strs.append(r_item.text() if r_item else "")
        s.angle_strs = angle_strs
        s.radius_strs = radius_strs
        self._redraw()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _redraw(self):
        theta_dir = 1 if self.dir_box.currentText() == "Counterclockwise" else -1
        self.canvas.redraw(
            series_list=self.series,
            theta_zero=self.zero_box.currentText(),
            theta_dir=theta_dir,
            r_max=self.rmax_spin.value(),
            show_grid=self.grid_chk.isChecked(),
            show_legend=self.legend_chk.isChecked(),
            title=self.title_edit.text(),
            style=self.style_box.currentText(),
        )

    def export_figure(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", "polar_plot.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if not path:
            return
        original_size = self.canvas.fig.get_size_inches()
        self.canvas.fig.set_size_inches(self.fig_width_spin.value(), self.fig_height_spin.value())
        try:
            self.canvas.fig.savefig(path, dpi=int(self.fig_dpi_spin.value()))
        finally:
            self.canvas.fig.set_size_inches(original_size)
            self.canvas.draw()


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
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(12, 12, 12, 12)

        btn_row = QHBoxLayout()
        btn_row2 = QHBoxLayout()
        load_btn = QPushButton("Load…")
        load_btn.clicked.connect(self.load_files)
        tooltip_bits = []
        if not SPE_AVAILABLE:
            tooltip_bits.append(
                "load_spe.py was not found next to this script — .spe loading is disabled."
            )
        if tooltip_bits:
            load_btn.setToolTip(" ".join(tooltip_bits))
        tools_btn = QPushButton("Curve Tools…")
        tools_btn.clicked.connect(self.open_curve_tools)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self.remove_selected)
        export_csv_btn = QPushButton("Export selected as CSV…")
        export_csv_btn.clicked.connect(self.export_selected_as_csv)
        polar_btn = QPushButton("Polar Plot…")
        polar_btn.clicked.connect(self.open_polar_plot)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(tools_btn)
        btn_row.addWidget(remove_btn)
        btn_row2.addWidget(export_csv_btn)
        btn_row2.addWidget(polar_btn)
        btn_row2.addStretch()
        left_layout.addLayout(btn_row)
        left_layout.addLayout(btn_row2)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        header = self.table.horizontalHeader()
        # Fixed, compact widths for every column except the last (Color),
        # which absorbs any extra space — keeps the table narrow enough to
        # avoid horizontal scrolling without relying on Qt's flakier
        # Stretch+Fixed mixing behavior.
        header.setSectionResizeMode(QHeaderView.Interactive)
        for col, width in [(0, 65), (1, 65), (2, 55), (3, 55), (4, 50), (5, 50)]:
            self.table.setColumnWidth(col, width)
        header.setMinimumSectionSize(30)
        header.setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.itemChanged.connect(self.on_table_item_changed)
        left_layout.addWidget(self.table)

        # ---- plot options ----
        opts = QGroupBox("Plot options")
        form = QFormLayout(opts)

        self.mode_box = ComboBox()
        self.mode_box.addItems(["Overlay", "Waterfall", "Power dependence"])
        self.mode_box.currentIndexChanged.connect(self.update_plot)
        form.addRow("Mode:", self.mode_box)

        self.offset_spin = make_spin(min_=-1e9, max_=1e9, decimals=6, value=1.0)
        self.offset_spin.valueChanged.connect(self.update_plot)
        form.addRow("Waterfall offset:", self.offset_spin)

        self.cmap_box = ComboBox()
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

        self.legend_loc_box = ComboBox()
        self.legend_loc_box.addItems([
            "best", "upper right", "upper left", "lower left", "lower right",
            "center left", "center right", "lower center", "upper center",
            "center", "outside right",
        ])
        self.legend_loc_box.currentIndexChanged.connect(self.update_plot)
        form.addRow("Legend position:", self.legend_loc_box)

        self.legend_fontsize_spin = make_spin(min_=4, max_=32, decimals=0, value=8)
        self.legend_fontsize_spin.valueChanged.connect(self.update_plot)
        form.addRow("Legend font size:", self.legend_fontsize_spin)

        self.legend_ncol_spin = make_spin(min_=1, max_=10, decimals=0, value=1)
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

        export_group, self.fig_width_spin, self.fig_height_spin, self.fig_dpi_spin = \
            make_export_group(self.export_figure, default_w=7.0, default_h=5.0)
        left_layout.addWidget(export_group)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        splitter.addWidget(left_scroll)

        # ---------------- right: plot ----------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)
        self.canvas = PlotCanvas()
        toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(toolbar)
        right_layout.addWidget(self.canvas)
        splitter.addWidget(right)

        splitter.setSizes([420, 880])

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select CSV or SPE files", "",
            "Supported files (*.csv *.txt *.spe);;CSV/Text files (*.csv *.txt);;SPE files (*.spe);;All files (*)"
        )
        if not paths:
            return
        for p in paths:
            if p.lower().endswith(".spe"):
                if not SPE_AVAILABLE:
                    QMessageBox.warning(
                        self, "SPE support unavailable",
                        "load_spe.py was not found next to this script.\n"
                        "Place load_spe.py in the same folder and restart the app."
                    )
                    continue
                try:
                    new_curves = self._curves_from_spe(p)
                except Exception as e:
                    QMessageBox.warning(self, "Load error", f"Could not read {p}:\n{e}")
                    continue
                self.curves.extend(new_curves)
            else:
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
        # Sniff delimiter; handles comma/semicolon/tab-delimited exports
        # (common from spectrometer software). Falls back to whitespace-
        # delimited parsing for .txt exports that use plain spaces, which
        # the sniffer can't reliably detect.
        with open(path, "r", newline="", errors="ignore") as f:
            sample = f.read(4096)
        try:
            dialect = csv_module.Sniffer().sniff(sample, delimiters=",;\t")
            df = pd.read_csv(path, sep=dialect.delimiter, engine="python")
            if df.shape[1] > 1:
                return df
        except csv_module.Error:
            pass
        # Whitespace fallback (handles single or repeated spaces/tabs)
        df = pd.read_csv(path, sep=r"\s+", engine="python")
        # If header row wasn't numeric-friendly and got parsed as data, pandas
        # already handles that via header inference; nothing more to do here.
        return df

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

    def _selected_curves(self):
        rows = selected_table_rows(self.table)
        return [self.curves[r] for r in rows if 0 <= r < len(self.curves)]

    def remove_selected(self):
        for r in selected_table_rows(self.table, reverse=True):
            if 0 <= r < len(self.curves):
                del self.curves[r]
        self.refresh_table()
        self.update_plot()

    def open_curve_tools(self):
        dialog = CurveToolsDialog(self, self._selected_curves(), self.curves, self)
        dialog.exec()
        self.refresh_table()
        self.update_plot()

    def open_polar_plot(self):
        dialog = PolarPlotDialog(self)
        dialog.exec()

    def export_selected_as_csv(self):
        selected = self._selected_curves()
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

            self.table.setCellWidget(row, 2, self._make_col_combo(c, "x_col"))
            self.table.setCellWidget(row, 3, self._make_col_combo(c, "y_col"))

            param_item = QTableWidgetItem(str(c.param))
            self.table.setItem(row, 4, param_item)

            self.table.setCellWidget(row, 5, self._make_visible_checkbox(c))

            color_btn = QPushButton()
            color_btn.setStyleSheet(f"background-color: {c.color};")
            color_btn.clicked.connect(lambda _, curve=c, btn=color_btn: self._pick_color(curve, btn))
            self.table.setCellWidget(row, 6, color_btn)

        self.table.blockSignals(False)

    def _make_visible_checkbox(self, curve):
        checkbox = QCheckBox()
        checkbox.setChecked(curve.visible)
        checkbox.toggled.connect(lambda checked, c=curve: self._set_visible(c, checked))
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(checkbox)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        return container

    def _set_visible(self, curve, checked):
        curve.visible = checked
        self.update_plot()

    def _make_col_combo(self, curve, attr):
        combo = ComboBox()
        combo.addItems([str(col) for col in curve.df.columns])
        combo.setCurrentText(str(getattr(curve, attr)))
        combo.currentTextChanged.connect(lambda text: self._set_col(curve, attr, text))
        # Without this, the combo sizes itself to its longest possible item
        # (e.g. "Wavelength (nm)"), forcing this column — and the whole
        # table — wider than necessary and triggering horizontal scrolling.
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(4)
        return combo

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
        if not path:
            return
        original_size = self.canvas.fig.get_size_inches()
        self.canvas.fig.set_size_inches(self.fig_width_spin.value(), self.fig_height_spin.value())
        try:
            self.canvas.fig.savefig(path, dpi=int(self.fig_dpi_spin.value()))
        finally:
            self.canvas.fig.set_size_inches(original_size)
            self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()