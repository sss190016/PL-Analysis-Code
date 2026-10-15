"""
Separation between every pair of fitted peaks, at each polarizer angle and
for H and V, in meV and cm^-1. Reads the peak_fits.csv written by
fit_polarization_spectra.py.

Usage:
  python peak_separations.py "C:\\path\\to\\exportedLineCuts"
     (the folder you ran fit_polarization_spectra.py on)

Outputs (in <folder>\\fit_results\\separations):
  - peak_separations.csv          one row per angle, polarization and peak pair
  - peak_separations_summary.csv  mean / std / min / max of each pair over angles
  - separation_<lo>-<hi>meV.png   one plot per peak pair
  - separations_vs_angle.png      all pairs side by side
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- settings
DATA_DIR = Path(r"/Users/seans/Documents/CdSe/082126/S2/bot/exportedLineCuts")  # folder containing the CSVs
#DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
FIT_CSV = DATA_DIR / "fit_results" / "peak_fits.csv"
OUT_DIR = DATA_DIR / "fit_results" / "separations"

MEV_TO_CM = 8.065544      # 1 meV = 8.065544 cm^-1
GROUP_TOL = 8.0           # meV: centres this close at different angles = same peak
ADJACENT_ONLY = True     # True: only neighbouring peaks (1-2, 2-3, ...)
POLARIZATIONS = ("H", "V")
MAKE_PLOT = True
COLORS = {"H": "blue", "V": "red"}
MARKERS = {"H": "s", "V": "p"}


# ------------------------------------------------------------------ helpers
def load_fits(path):
    if not path.exists():
        sys.exit(f"Can't find {path}\nRun fit_polarization_spectra.py on this "
                 f"folder first.")
    return pd.read_csv(path)


def assign_groups(df):
    """Give every fitted peak a label (its group's typical centre) so the same
    peak is recognised at every angle even if its centre moves a little."""
    c = np.sort(df["center"].to_numpy(float))
    groups, cur = [], [c[0]]
    for v in c[1:]:
        if v - np.median(cur) <= GROUP_TOL:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)
    labels = np.array([np.median(g) for g in groups])
    idx = np.abs(df["center"].to_numpy(float)[:, None] - labels[None, :]).argmin(1)
    df = df.copy()
    df["peak_group"] = np.round(labels[idx], 1)
    return df, labels


def separations(df):
    rows = []
    for (x, rot, pol), g in df.groupby(["x_index", "rotation_deg", "polarization"]):
        g = g.sort_values("center")
        pairs = (zip(range(len(g) - 1), range(1, len(g))) if ADJACENT_ONLY
                 else combinations(range(len(g)), 2))
        for i, j in pairs:
            a, b = g.iloc[i], g.iloc[j]
            sep = b["center"] - a["center"]
            err = np.hypot(a.get("center_err", np.nan), b.get("center_err", np.nan))
            rows.append({
                "x_index": x, "rotation_deg": rot, "polarization": pol,
                "pair": f"{a['peak_group']:.1f}-{b['peak_group']:.1f}",
                "lower_peak_meV": a["center"], "upper_peak_meV": b["center"],
                "separation_meV": sep, "separation_err_meV": err,
                "separation_cm-1": sep * MEV_TO_CM,
                "separation_err_cm-1": err * MEV_TO_CM,
            })
    return pd.DataFrame(rows)


def summarise(sep):
    g = sep.groupby(["pair", "polarization"])["separation_meV"]
    s = g.agg(n_angles="count", mean_meV="mean", std_meV="std",
              min_meV="min", max_meV="max").reset_index()
    for k in ("mean", "std", "min", "max"):
        s[f"{k}_cm-1"] = s[f"{k}_meV"] * MEV_TO_CM
    # pair order: by lower then upper peak energy
    s["_o"] = s["pair"].str.split("-").apply(lambda p: (float(p[0]), float(p[1])))
    return s.sort_values(["_o", "polarization"]).drop(columns="_o")


def sorted_pairs(sep):
    return sorted(sep["pair"].unique(),
                  key=lambda p: tuple(float(v) for v in p.split("-")))


def draw_pair(ax, sep, p):
    d = sep[sep["pair"] == p]
    for pol in POLARIZATIONS:
        e = d[d["polarization"] == pol].sort_values("rotation_deg")
        ax.plot(e["rotation_deg"], e["separation_meV"], "-",
                marker=MARKERS[pol], color=COLORS[pol], ms=5, lw=1.2,
                label=pol)
    lo, hi = p.split("-")
    ax.set_title(f"{lo} ↔ {hi} meV", fontsize=11)
    ax.set_xlabel("Rotation angle (deg)")
    ax.set_ylabel("Separation (meV)")
    ax.grid(True, color="0.85", lw=0.6)
    ax.legend(fontsize=8, loc="best")
    # second y axis in cm^-1
    sec = ax.secondary_yaxis("right", functions=(lambda v: v * MEV_TO_CM,
                                                 lambda v: v / MEV_TO_CM))
    sec.set_ylabel("Separation (cm$^{-1}$)")


def plot_each(sep, out_dir):
    """One figure per peak pair, like the individual polar plots."""
    paths = []
    for p in sorted_pairs(sep):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        draw_pair(ax, sep, p)
        fig.tight_layout()
        lo, hi = p.split("-")
        path = out_dir / f"separation_{float(lo):.0f}-{float(hi):.0f}meV.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def plot(sep, path):
    pairs = sorted_pairs(sep)
    n = len(pairs)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow),
                            squeeze=False)
    for ax, p in zip(axs.flat, pairs):
        draw_pair(ax, sep, p)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------- main
def main():
    df = load_fits(FIT_CSV)
    df = df[df["polarization"].isin(POLARIZATIONS)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, labels = assign_groups(df)
    print(f"Read {len(df)} fitted peaks from {df['x_index'].nunique()} angles; "
          f"peak groups at {', '.join(f'{v:.1f}' for v in labels)} meV")

    sep = separations(df)
    if sep.empty:
        sys.exit("No angle has two or more peaks - nothing to separate.")
    sep = sep.sort_values(["rotation_deg", "polarization", "lower_peak_meV",
                           "upper_peak_meV"])
    sep.to_csv(OUT_DIR / "peak_separations.csv", index=False)

    summ = summarise(sep)
    summ.to_csv(OUT_DIR / "peak_separations_summary.csv", index=False)

    n_ang = df["x_index"].nunique()
    print("\nPair (meV)          pol  angles   mean meV ± std     mean cm-1 ± std")
    for _, r in summ.iterrows():
        print(f"  {r['pair']:<18} {r['polarization']:>2}   {r['n_angles']:>2}/{n_ang}"
              f"   {r['mean_meV']:7.2f} ± {r['std_meV']:5.2f}"
              f"   {r['mean_cm-1']:8.1f} ± {r['std_cm-1']:5.1f}")

    n_fig = 0
    if MAKE_PLOT:
        n_fig = len(plot_each(sep, OUT_DIR))
        plot(sep, OUT_DIR / "separations_vs_angle.png")
    print(f"\nSaved peak_separations.csv, peak_separations_summary.csv"
          + (f", {n_fig} individual separation plot(s) and "
             f"separations_vs_angle.png" if MAKE_PLOT else "")
          + f" to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()