"""
spe_converter.py

Batch-convert every .spe file in a folder to .csv, one CSV per SPE file.

Usage
-----
    python spe_converter.py              # pick a folder with a dialog
    python spe_converter.py /path/to/dir # or pass the folder directly

CSVs are written to a "csvs" subfolder of the selected folder (created if
needed), each with the same base name as its source file.
The first column is wavelength (nm, or pixel index if the file has no
calibration); the remaining columns are intensity, one per row/frame:
    - single row, single frame  -> "Intensity"
    - several rows / frames     -> "Row 1", "Frame 1 Row 1", ...
Multi-ROI (v3.x) files are written as one CSV per ROI (<name>_ROI1.csv, ...).
"""

import os
import sys

import numpy as np

from load_spe import load_spe


def pick_folder():
    """Open a native folder-picker dialog; returns None if cancelled."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(title="Select a folder containing .spe files")
    root.destroy()
    return path or None


def _columns(arr):
    """Flatten a (width, height[, frames]) array into named intensity columns."""
    if arr.ndim == 2:
        width, height = arr.shape
        if height == 1:
            return ["Intensity"], arr.reshape(width, 1)
        return [f"Row {r + 1}" for r in range(height)], arr

    width, height, n_frames = arr.shape
    names, cols = [], []
    for f in range(n_frames):
        for r in range(height):
            names.append(f"Frame {f + 1}" if height == 1 else f"Frame {f + 1} Row {r + 1}")
            cols.append(arr[:, r, f])
    return names, np.column_stack(cols)


def _write_csv(out_path, wavelengths, arr):
    names, values = _columns(arr)
    wavelengths = np.asarray(wavelengths, dtype=float)
    if wavelengths.size != values.shape[0]:
        # Missing/mismatched calibration -- fall back to pixel index
        wavelengths = np.arange(1, values.shape[0] + 1, dtype=float)
    table = np.column_stack([wavelengths, values.astype(float)])
    header = ",".join(["Wavelength (nm)"] + names)
    np.savetxt(out_path, table, delimiter=",", header=header, comments="", fmt="%.6g")


def convert_file(spe_path, out_dir):
    """Convert one .spe file into out_dir; returns the list of CSV paths written."""
    data, wavelengths, _ = load_spe(spe_path)
    base = os.path.join(out_dir, os.path.splitext(os.path.basename(spe_path))[0])

    if isinstance(data, np.ndarray):
        out = base + ".csv"
        _write_csv(out, wavelengths, data)
        return [out]

    # Multi-ROI: dict {'ROI': [...]} for one frame, list of such dicts for many
    frames = data if isinstance(data, list) else [data]
    n_roi = len(frames[0]['ROI'])
    written = []
    for i in range(n_roi):
        roi_stack = [fr['ROI'][i] for fr in frames]
        arr = roi_stack[0] if len(roi_stack) == 1 else np.stack(roi_stack, axis=2)
        wl = wavelengths[i] if isinstance(wavelengths, list) else wavelengths
        out = f"{base}_ROI{i + 1}.csv"
        _write_csv(out, wl, arr)
        written.append(out)
    return written


def convert_folder(folder):
    spe_files = sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(".spe") and not f.startswith(".")
    )
    if not spe_files:
        print(f"No .spe files found in {folder}")
        return

    out_dir = os.path.join(folder, "csvs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Converting {len(spe_files)} file(s) in {folder} -> {out_dir}")
    ok = 0
    for path in spe_files:
        name = os.path.basename(path)
        try:
            outputs = convert_file(path, out_dir)
            ok += 1
            print(f"  {name} -> {', '.join(os.path.basename(o) for o in outputs)}")
        except Exception as e:
            print(f"  {name} FAILED: {e}")
    print(f"Done: {ok}/{len(spe_files)} converted.")


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else pick_folder()
    if not folder:
        print("No folder selected.")
        sys.exit(0)
    convert_folder(folder)
