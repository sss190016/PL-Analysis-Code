"""
Polar plots of peak area vs. polarizer rotation angle, built from the
peak_fits.csv written by fit_polarization_spectra.py.

The script groups peaks from different angles by centre energy, keeps the
peaks that appear in (nearly) every data set, and makes one polar plot per
peak with the H and V areas as two series.

Usage:
  python polarization_plots.py "C:\\path\\to\\exportedLineCuts"
     (the folder you ran fit_polarization_spectra.py on; the script reads
      its fit_results\\peak_fits.csv)

Outputs (in <folder>\\fit_results\\polar_plots):
  - polar_peak_<centre>meV.png   one polar plot per peak
  - polar_all_peaks.png          all tracked peaks side by side
  - polar_peak_areas.csv         area table: one row per angle, per peak
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- settings
DATA_DIR = Path(r"C:\Users\seans\Documents\CdSe\082126\S2\top\exportedLineCuts")
#DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
FIT_CSV = DATA_DIR / "fit_results" / "peak_fits.csv"
OUT_DIR = DATA_DIR / "fit_results" / "polar_plots"

# Which peaks to plot.
#   None  -> automatic: group peaks by centre and keep the ones found in at
#            least MIN_PRESENCE of the angles.
#   list  -> plot exactly these centres (meV), e.g. [1856, 1882]
PEAK_CENTERS = None
GROUP_TOL = 12.0        # peaks within this many meV of each other = same peak
MIN_PRESENCE = 1.0     # 1.0 = must appear at every angle; 0.8 = 80 % of them

POLARIZATIONS = ("H", "V")          # which series to draw
LABELS = {"H": "H", "V": "V"}          # legend text in "fraction" mode
RAW_LABELS = {"H": "H", "V": "V"}      # legend text in "area"/"max" modes
COLORS = {"H": "blue", "V": "red"}
MARKERS = {"H": "s", "V": "p"}      # square, pentagon (as in your example)

# What goes on the radial axis:
#   "fraction" -> H/(H+V) and V/(H+V) at each angle (0..1, sum = 1)
#   "area"     -> raw fitted peak areas
#   "max"      -> each series divided by its own maximum
PLOT_MODE = "fraction"
SHOW_ERRORS = False    # True: draw +/- 1 sigma area error bars
R_MAX = None           # fixed radial limit, or None for automatic
RADIAL_LABEL = "Peak area (a.u.)"
SIDE_AXIS = True       # vertical intensity scale on the left, like your plot
CLOSE_LOOP = True      # join the last angle back to the first


# ------------------------------------------------------------------ helpers
def load_fits(path):
    if not path.exists():
        sys.exit(f"Can't find {path}\nRun fit_polarization_spectra.py on this "
                 f"folder first.")
    df = pd.read_csv(path)
    df["center"] = df["center"].astype(float)
    return df


def group_centres(df):
    """Cluster all fitted centres; return the cluster centres (meV) that
    show up at enough angles."""
    n_angles = df["x_index"].nunique()
    # use H centres to define groups (the V centre can be shifted slightly)
    c = np.sort(df.loc[df["polarization"] == "H", "center"].to_numpy())
    groups, cur = [], [c[0]]
    for v in c[1:]:
        if v - np.median(cur) <= GROUP_TOL:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)

    keep = []
    for g in groups:
        centre = float(np.median(g))
        sub = df[(df["polarization"] == "H")
                 & (np.abs(df["center"] - centre) <= GROUP_TOL)]
        present = sub["x_index"].nunique()
        tag = "kept" if present >= MIN_PRESENCE * n_angles else "skipped"
        print(f"  peak near {centre:7.1f} meV: found at {present}/{n_angles} "
              f"angles -> {tag}")
        if tag == "kept":
            keep.append(centre)
    return keep


def track_peak(df, centre):
    """For each angle and polarization, pick the fitted peak closest to
    `centre` (within GROUP_TOL). Missing ones become NaN."""
    rows = []
    for (x, rot), grp in df.groupby(["x_index", "rotation_deg"]):
        row = {"x_index": x, "rotation_deg": rot}
        for pol in ("H", "V"):
            g = grp[grp["polarization"] == pol]
            d = np.abs(g["center"] - centre)
            if len(g) and d.min() <= GROUP_TOL:
                r = g.loc[d.idxmin()]
                row[f"center_{pol}"] = r["center"]
                row[f"fwhm_{pol}"] = r["fwhm"]
                row[f"area_{pol}"] = r["area"]
                row[f"area_err_{pol}"] = r["area_err"]
            else:
                for k in ("center", "fwhm", "area", "area_err"):
                    row[f"{k}_{pol}"] = np.nan
        rows.append(row)
    t = pd.DataFrame(rows).sort_values("rotation_deg").reset_index(drop=True)
    tot = t["area_H"] + t["area_V"]
    t["frac_H_H/(H+V)"] = t["area_H"] / tot
    t["frac_V_V/(H+V)"] = t["area_V"] / tot
    t["DOLP_(H-V)/(H+V)"] = (t["area_H"] - t["area_V"]) / tot
    return t


def series(t, pol):
    th = np.deg2rad(t["rotation_deg"].to_numpy(float))
    r = t[f"area_{pol}"].to_numpy(float)
    e = t[f"area_err_{pol}"].to_numpy(float)
    if PLOT_MODE == "fraction":
        other = "V" if pol == "H" else "H"
        o = t[f"area_{other}"].to_numpy(float)
        eo = t[f"area_err_{other}"].to_numpy(float)
        tot = r + o
        # error of r/(r+o):  d/dr = o/tot^2,  d/do = -r/tot^2
        e = np.sqrt((o * e) ** 2 + (r * eo) ** 2) / tot ** 2
        r = r / tot
    elif PLOT_MODE == "max" and np.nanmax(r) > 0:
        s = np.nanmax(r)
        r, e = r / s, e / s
    if CLOSE_LOOP and len(th) > 1 and not np.isclose(
            np.mod(np.rad2deg(th[-1]) - np.rad2deg(th[0]), 360), 0):
        th, r, e = np.append(th, th[0] + 2 * np.pi), np.append(r, r[0]), np.append(e, e[0])
    return th, r, e


def draw_polar(ax, t, title):
    rmax = 0.0
    for pol in POLARIZATIONS:
        th, r, e = series(t, pol)
        lab = LABELS[pol] if PLOT_MODE == "fraction" else RAW_LABELS[pol]
        ax.plot(th, r, "-", marker=MARKERS[pol], color=COLORS[pol], ms=5,
                lw=1.4, label=lab)
        if SHOW_ERRORS:
            ax.errorbar(th, r, yerr=e, fmt="none", ecolor=COLORS[pol],
                        elinewidth=0.8, capsize=2)
        rmax = max(rmax, np.nanmax(r + (e if SHOW_ERRORS else 0)))
    rlim = R_MAX or (1.0 if PLOT_MODE == "fraction"
                     else 1.05 if PLOT_MODE == "max" else rmax * 1.1)
    ax.set_ylim(0, rlim)
    ax.set_thetagrids(np.arange(0, 360, 20), [f"{a}°" for a in range(0, 360, 20)])
    ax.set_yticklabels([])                 # scale is shown on the side axis
    ax.grid(True, color="0.7", lw=0.6)
    ax.set_title(title, y=1.17, fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2,
              fontsize=9, frameon=True)
    ax.set_xlabel("Rotation angle (deg)", labelpad=14)
    return rlim


def radial_label():
    if PLOT_MODE == "fraction":
        return "Polarization fraction  I/(I$_H$+I$_V$)"
    if PLOT_MODE == "max":
        return "Normalized " + RADIAL_LABEL
    return RADIAL_LABEL


def add_side_axis(fig, ax, rlim):
    """Vertical scale on the left running -rmax..rmax through the centre,
    shown as positive numbers both ways (like the example plot)."""
    pos = ax.get_position()
    side = fig.add_axes([pos.x0 - 0.09, pos.y0, 0.001, pos.height])
    side.set_ylim(-rlim, rlim)
    ticks = side.get_yticks()
    ticks = ticks[(ticks >= -rlim) & (ticks <= rlim)]
    side.set_yticks(ticks)
    side.set_yticklabels([f"{abs(v):g}" for v in ticks])
    side.set_xticks([])
    for sp in ("top", "right", "bottom"):
        side.spines[sp].set_visible(False)
    side.set_ylabel(radial_label(), fontsize=11)


def plot_one(t, centre, path):
    fig = plt.figure(figsize=(6.4, 6))
    ax = fig.add_axes([0.2, 0.1, 0.72, 0.72], projection="polar")
    rlim = draw_polar(ax, t, f"Peak at {centre:.1f} meV")
    if SIDE_AXIS:
        add_side_axis(fig, ax, rlim)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_all(tables, path):
    n = len(tables)
    fig = plt.figure(figsize=(5.4 * n, 6.2))
    for i, (centre, t) in enumerate(tables.items()):
        ax = fig.add_subplot(1, n, i + 1, projection="polar")
        draw_polar(ax, t, f"Peak at {centre:.1f} meV")
        rl = ax.get_ylim()[1]                # show scale on the polar axis
        ticks = plt.MaxNLocator(4).tick_values(0, rl)
        ticks = ticks[(ticks > 0) & (ticks <= rl)]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:,.0f}" if v >= 10 else f"{v:.2g}" for v in ticks],
                           fontsize=7, color="0.35")
        ax.set_rlabel_position(70)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.8, bottom=0.08, wspace=0.35)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------- main
def main():
    df = load_fits(FIT_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Read {len(df)} fitted peaks from {df['x_index'].nunique()} angles")

    if PEAK_CENTERS is None:
        centres = group_centres(df)
    else:
        centres = [float(c) for c in PEAK_CENTERS]
    if not centres:
        sys.exit("No peak appears at enough angles. Lower MIN_PRESENCE or set "
                 "PEAK_CENTERS by hand.")

    tables, out = {}, []
    for c in centres:
        t = track_peak(df, c)
        miss = t["area_H"].isna().sum()
        if miss:
            print(f"  note: peak {c:.1f} meV missing at {miss} angle(s) - gaps")
        tables[c] = t
        plot_one(t, c, OUT_DIR / f"polar_peak_{c:.0f}meV.png")
        out.append(t.assign(peak_meV=round(c, 1)))

    if len(tables) > 1:
        plot_all(tables, OUT_DIR / "polar_all_peaks.png")
    cols = ["peak_meV", "x_index", "rotation_deg", "area_H", "area_err_H",
            "area_V", "area_err_V", "frac_H_H/(H+V)", "frac_V_V/(H+V)",
            "DOLP_(H-V)/(H+V)", "center_H", "center_V",
            "fwhm_H", "fwhm_V"]
    pd.concat(out)[cols].to_csv(OUT_DIR / "polar_peak_areas.csv", index=False)
    print(f"\nSaved {len(tables)} polar plot(s) and polar_peak_areas.csv to "
          f"{OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()