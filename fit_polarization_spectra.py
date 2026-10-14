"""
Plot H/V polarization pairs and fit 2-5 peaks to each pair.

Files: S1_<x>.spe<y>LineEV.csv   (e.g. S1_0.spe53LineEV.csv)
  x  -> polarizer index (rotation angle = 20 * x degrees)
  y  -> the smaller y of each pair = horizontal (H), the larger = vertical (V)

H and V are fitted TOGETHER: every peak has one centre (and, by default, one
width) shared by both spectra, plus its own area in H and in V. So a weak V
spectrum is fitted at the same peak positions as its H partner.

Outputs (in OUTPUT_DIR):
  - one PNG per angle (H and V data, fits, individual peaks, residuals)
  - peak_fits.csv  : centre, FWHM, area (+ 1-sigma errors) for every peak,
                     one row per peak per polarization
  - fit_summary.csv: number of peaks chosen, R^2 (H and V), BIC per pair

Usage:
  python fit_polarization_spectra.py "C:\\path\\to\\exportedLineCuts"
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.special import voigt_profile

# ----------------------------------------------------------------- settings
DATA_DIR = Path(r"C:\Users\seans\Documents\CdSe\082126\S2\top\exportedLineCuts")
#DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUTPUT_DIR = DATA_DIR / "fit_results"
FILE_REGEX = r"S\d+_(\d+)[._]spe\D*(\d+).*\.csv$"   # e.g. S1_0.spe53LineEV.csv
ANGLE_STEP = 20                                   # degrees per x step

PEAK_SHAPE = "voigt"    # "gaussian", "lorentzian", or "voigt"
MIN_PEAKS, MAX_PEAKS = 2, 5
BACKGROUND = "linear"      # "constant", "linear", or "none" (separate for H, V)
SHARE_WIDTHS = False        # True: a peak has the same FWHM in H and V
MAX_SHIFT = 5.0            # V peak centre may sit up to +/- this far (meV)
                           # from the H centre. 0 = identical positions.
X_RANGE = None             # e.g. (1750, 1950) to fit only part of the spectrum
X_LABEL = "Energy (meV)"
Y_LABEL = "Intensity (counts)"

# Narrowest / widest peak allowed, in x-axis units (meV).
# MIN_FWHM stops cosmic-ray spikes being fitted; MAX_FWHM forbids broad
# pedestal-like peaks. Adjust if your real lines fall outside this range.
MIN_FWHM = 5.0
MAX_FWHM = 35.0

# Two peaks must be at least this far apart (meV). Stops one asymmetric
# line being modelled as two Gaussians stacked on top of each other.
MIN_SEPARATION = 10.0

# Force the peak count for specific angles, e.g. {3: 4, 18: 5}  (keys = x).
# Anything not listed is chosen automatically by BIC.
FORCE_N_PEAKS = {}

BIC_MARGIN = 10.0     # extra peak kept only if it lowers BIC by more than this
MIN_SNR = 3.0         # a peak is kept only if area/area_error >= this in H or V
EDGE_MARGIN = 0.03    # peak centres kept this fraction of the range from edges

POLS = ("H", "V")


# ----------------------------------------------------------- peak shapes
# Single-peak functions take (x, centre, area, width[, width2]).
def _gauss(x, c, a, s):
    return a * np.exp(-0.5 * ((x - c) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def _lorentz(x, c, a, g):
    return a * (g / np.pi) / ((x - c) ** 2 + g ** 2)


def _voigt(x, c, a, s, g):
    return a * voigt_profile(x - c, s, g)


SHAPES = {"gaussian": (_gauss, 1), "lorentzian": (_lorentz, 1), "voigt": (_voigt, 2)}
PEAK_FN, NW = SHAPES[PEAK_SHAPE]                 # NW = width params per peak
NBG = {"none": 0, "constant": 1, "linear": 2}[BACKGROUND]
NWJ = NW if SHARE_WIDTHS else 2 * NW             # width params per joint peak
NPJ = 4 + NWJ                                    # c, aH, aV, widths..., dV


def fwhm_from_widths(w):
    if PEAK_SHAPE == "gaussian":
        return 2 * np.sqrt(2 * np.log(2)) * w[0]
    if PEAK_SHAPE == "lorentzian":
        return 2 * w[0]
    fg, fl = 2 * np.sqrt(2 * np.log(2)) * w[0], 2 * w[1]   # Olivero approx.
    return 0.5346 * fl + np.sqrt(0.2166 * fl ** 2 + fg ** 2)


def widths_from_fwhm(f):
    if PEAK_SHAPE == "gaussian":
        return [f / 2.3548]
    if PEAK_SHAPE == "lorentzian":
        return [f / 2]
    return [f / 2.3548 / 1.5, f / 2 / 1.5]


# ------------------------------------------------- joint parameter layout
# p = [bg_H (NBG), bg_V (NBG), peak1 (NPJ), peak2 (NPJ), ...]
# peak = [centre_H, area_H, area_V, widths_H(, widths_V), shift_V]
#   (V centre = centre_H + shift_V, |shift_V| <= MAX_SHIFT)
def peak_slice(i):
    return slice(2 * NBG + i * NPJ, 2 * NBG + (i + 1) * NPJ)


def single_params(pk, pol):
    """Joint peak params -> (centre, area, *widths) for one polarization."""
    k = POLS.index(pol)
    if SHARE_WIDTHS:
        w = pk[3:3 + NW]
    else:
        w = pk[3 + k * NW: 3 + (k + 1) * NW]
    c = pk[0] + (pk[-1] if pol == "V" else 0.0)
    return np.concatenate([[c, pk[1 + k]], w])


def centre_matrix(n, npar):
    """Linear map raw params -> absolute params. In the raw vector the first
    peak stores its centre and every later peak stores the GAP to the previous
    centre (gap >= MIN_SEPARATION), so peaks can never sit on top of each other."""
    A = np.eye(npar)
    for i in range(n):
        for j in range(i):
            A[peak_slice(i).start, peak_slice(j).start] = 1.0
    return A


def to_absolute(p, n):
    p = np.array(p, float)
    idx = [peak_slice(i).start for i in range(n)]
    p[idx] = np.cumsum(p[idx])
    return p


def background(x, b, x0):
    if NBG == 0:
        return np.zeros_like(x)
    if NBG == 1:
        return np.full_like(x, b[0])
    return b[0] + b[1] * (x - x0)


def eval_pol(x, p, n, pol, x0):
    k = POLS.index(pol)
    y = background(x, p[k * NBG:(k + 1) * NBG], x0)
    for i in range(n):
        y = y + PEAK_FN(x, *single_params(p[peak_slice(i)], pol))
    return y


# --------------------------------------------------------------- loading
def load_spectrum(path):
    """Read the first two numeric columns, with or without a header row."""
    df = pd.read_csv(path, header=None, sep=None, engine="python")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all", axis=1)
    df = df.dropna()
    x, y = df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if X_RANGE is not None:
        m = (x >= X_RANGE[0]) & (x <= X_RANGE[1])
        x, y = x[m], y[m]
    return x, y


def find_pairs(folder):
    groups = {}
    for f in sorted(folder.glob("*.csv")):
        m = re.search(FILE_REGEX, f.name, re.IGNORECASE)
        if m:
            groups.setdefault(int(m.group(1)), []).append((int(m.group(2)), f))
    pairs = {}
    for x, files in sorted(groups.items()):
        if len(files) != 2:
            print(f"  ! x={x}: expected 2 files, found {len(files)} - skipped")
            continue
        files.sort()                              # smaller y first -> H
        pairs[x] = {"H": files[0][1], "V": files[1][1]}
    return pairs


# ---------------------------------------------------------------- fitting
def noise_level(y):
    """Point-to-point noise estimate (robust to peaks and slopes)."""
    d = np.diff(y)
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)


def smooth(y, n):
    win = max(5, (n // 50) | 1)
    return savgol_filter(y, win, 2) if len(y) > win else y


class Pair:
    """H and V data prepared for a joint fit."""

    def __init__(self, xH, yH, xV, yV):
        self.d = {"H": (xH, yH), "V": (xV, yV)}
        self.sig = {p: max(noise_level(self.d[p][1]), 1e-9) for p in POLS}
        self.x0 = np.mean(np.concatenate([xH, xV]))
        self.xmin = max(xH.min(), xV.min())
        self.xmax = min(xH.max(), xV.max())
        self.span = self.xmax - self.xmin
        self.dx = np.median(np.diff(xH))
        self.X = np.concatenate([xH, xV])
        self.Y = np.concatenate([yH, yV])
        self.S = np.concatenate([np.full(len(xH), self.sig["H"]),
                                 np.full(len(xV), self.sig["V"])])
        self.nH = len(xH)
        # smoothed, baseline-ish-removed copies for guessing heights
        self.sm = {}
        for p in POLS:
            x, y = self.d[p]
            ys = smooth(y, len(y))
            self.sm[p] = (x, ys - np.percentile(ys, 10))

    def height_at(self, pol, c):
        x, ys = self.sm[pol]
        return max(np.interp(c, x, ys), 1e-9)

    def model(self, n):
        nH = self.nH

        def f(X, *p):
            p = to_absolute(p, n)
            return np.concatenate([eval_pol(X[:nH], p, n, "H", self.x0),
                                   eval_pol(X[nH:], p, n, "V", self.x0)])
        return f


def candidate_peaks(pr, k=8):
    """Peak positions from H+V combined (each normalised by its noise)."""
    xH, _ = pr.d["H"]
    comb = pr.sm["H"][1] / pr.sig["H"] + np.interp(xH, *pr.sm["V"]) / pr.sig["V"]
    pk, props = find_peaks(comb, prominence=0.05 * (comb.max() - comb.min()),
                           width=max(1, 0.5 * MIN_FWHM / pr.dx))
    order = np.argsort(props["prominences"])[::-1][:k]
    return [(xH[pk[i]], props["widths"][i] * pr.dx) for i in order]


def fit_joint(pr, guesses):
    """guesses: list of (centre, fwhm). Returns result dict."""
    n = len(guesses)
    wmin, wmax = max(MIN_FWHM, 2 * pr.dx), MAX_FWHM
    cmin = pr.xmin + EDGE_MARGIN * pr.span
    cmax = pr.xmax - EDGE_MARGIN * pr.span

    p0, lo, hi = [], [], []
    for pol in POLS:
        y = pr.d[pol][1]
        yr = y.max() - y.min()
        if NBG >= 1:
            p0 += [np.percentile(y, 10)]; lo += [y.min() - yr]; hi += [y.max()]
        if NBG == 2:
            lim = 5 * yr / pr.span
            p0 += [0.0]; lo += [-lim]; hi += [lim]

    wl, wh = widths_from_fwhm(wmin), widths_from_fwhm(wmax)
    if PEAK_SHAPE == "voigt":                # allow either component to shrink
        wl = [wl[0] / 3, wl[1] / 3]
    for c, f in guesses:
        c = float(np.clip(c, cmin, cmax))
        f = float(np.clip(f, 1.2 * wmin, 0.8 * wmax))
        w = widths_from_fwhm(f)
        areas = [pr.height_at(pol, c) * f * 1.06 for pol in POLS]
        wset = w if SHARE_WIDTHS else w + w
        ms = max(MAX_SHIFT, 1e-6)
        p0 += [c] + areas + wset + [0.0]
        lo += [cmin, 0, 0] + (wl if SHARE_WIDTHS else wl + wl) + [-ms]
        hi += [cmax, 1e15, 1e15] + (wh if SHARE_WIDTHS else wh + wh) + [ms]

    # convert centres to (first centre, gaps) form
    order = np.argsort([p0[peak_slice(i).start] for i in range(n)])
    blocks = [(p0[peak_slice(i)], lo[peak_slice(i)], hi[peak_slice(i)]) for i in order]
    head = 2 * NBG
    p0, lo, hi = p0[:head], lo[:head], hi[:head]
    prev = None
    for b0, bl, bh in blocks:
        b0, bl, bh = list(b0), list(bl), list(bh)
        if prev is not None:
            gap = max(b0[0] - prev, MIN_SEPARATION * 1.05)
            prev = prev + gap
            b0[0], bl[0], bh[0] = gap, MIN_SEPARATION, pr.span
        else:
            prev = b0[0]
        p0, lo, hi = p0 + b0, lo + bl, hi + bh
    lo, hi = np.array(lo, float), np.array(hi, float)
    p0 = np.clip(np.array(p0, float), lo + 1e-9 * (np.abs(lo) + 1),
                 hi - 1e-9 * (np.abs(hi) + 1))
    model = pr.model(n)
    raw, rcov = curve_fit(model, pr.X, pr.Y, p0=p0, sigma=pr.S,
                          bounds=(lo, hi), maxfev=4000)
    resid = pr.Y - model(pr.X, *raw)
    if to_absolute(raw, n)[peak_slice(n - 1).start] > pr.xmax:
        raise RuntimeError("peak beyond data")
    A = centre_matrix(n, len(raw))
    popt, pcov = to_absolute(raw, n), A @ rcov @ A.T
    chi = resid / pr.S
    N = len(pr.X)
    bic = N * np.log(np.sum(chi ** 2) / N) + len(raw) * np.log(N)
    return dict(n=n, popt=popt, pcov=pcov, bic=bic,
                resid={"H": resid[:pr.nH], "V": resid[pr.nH:]})


def significant(res):
    """Each peak must be significant (area >= MIN_SNR * error) in H or V,
    and no two peaks may be closer than MIN_SEPARATION."""
    cs = np.sort([res["popt"][peak_slice(i)][0] for i in range(res["n"])])
    if len(cs) > 1 and np.min(np.diff(cs)) < MIN_SEPARATION:
        return False
    perr = np.sqrt(np.clip(np.diag(res["pcov"]), 0, None))
    for i in range(res["n"]):
        sl = peak_slice(i)
        a, e = res["popt"][sl][1:3], perr[sl][1:3]
        if not np.any(np.isfinite(e) & (a >= MIN_SNR * e) & (a > 0)):
            return False
    return True


def current_peaks(res):
    out = []
    for i in range(res["n"]):
        pk = res["popt"][peak_slice(i)]
        out.append((pk[0], fwhm_from_widths(pk[3:3 + NW])))
    return out


def residual_peak(pr, res):
    """Biggest leftover bump in (H + V) residuals, noise-weighted."""
    xH, _ = pr.d["H"]
    r = (smooth(res["resid"]["H"], pr.nH) / pr.sig["H"]
         + np.interp(xH, pr.d["V"][0],
                     smooth(res["resid"]["V"], len(res["resid"]["V"]))) / pr.sig["V"])
    for c, _ in current_peaks(res):          # only look away from existing peaks
        r[np.abs(xH - c) < MIN_SEPARATION] = -np.inf
    i = int(np.argmax(r))
    return (xH[i], 2 * MIN_FWHM)


def fit_pair(pr, forced_n=None):
    cands = candidate_peaks(pr)
    if not cands:
        xH, yH = pr.d["H"]
        cands = [(xH[np.argmax(smooth(yH, len(yH)))], 3 * MIN_FWHM)]

    def attempt(g):
        try:
            return fit_joint(pr, g)
        except (RuntimeError, ValueError):
            return None

    n_max = forced_n or MAX_PEAKS
    results, prev = {}, None
    for n in range(1, n_max + 1):
        seeds = []
        if len(cands) >= n:
            seeds.append(cands[:n])
        if prev is not None:
            base = current_peaks(prev)
            seeds.append(base + [residual_peak(pr, prev)])
            unused = [g for g in cands
                      if all(abs(g[0] - b[0]) > MIN_SEPARATION for b in base)]
            if unused:
                seeds.append(base + [unused[0]])
        fits = [f for f in (attempt(s) for s in seeds if len(s) == n) if f]
        if not fits:
            break
        good = [f for f in fits if significant(f)]
        best = min(good or fits, key=lambda f: f["bic"])
        best["significant"] = bool(good)
        results[n] = best
        prev = best

    if not results:
        raise RuntimeError("no fit converged")
    if forced_n:
        return results.get(forced_n, results[max(results)])
    ok = {n: r for n, r in results.items() if r["significant"]}
    pool = {n: r for n, r in ok.items() if n >= MIN_PEAKS} or ok or results
    best_bic = min(r["bic"] for r in pool.values())
    for n in sorted(pool):           # fewest peaks within BIC_MARGIN of best
        if pool[n]["bic"] <= best_bic + BIC_MARGIN:
            return pool[n]


def r_squared(pr, res, pol):
    y = pr.d[pol][1]
    return 1 - np.sum(res["resid"][pol] ** 2) / np.sum((y - y.mean()) ** 2)


def peak_rows(res):
    """One row per peak per polarization, with propagated FWHM error."""
    popt, pcov = res["popt"], res["pcov"]
    perr = (np.sqrt(np.clip(np.diag(pcov), 0, None))
            if np.all(np.isfinite(pcov)) else np.full_like(popt, np.nan))
    rows = []
    for i in range(res["n"]):
        sl = peak_slice(i)
        pk, ek = popt[sl], perr[sl]
        for k, pol in enumerate(POLS):
            wi = slice(3, 3 + NW) if SHARE_WIDTHS else slice(3 + k * NW, 3 + (k + 1) * NW)
            w, we = pk[wi], ek[wi]
            f = fwhm_from_widths(w)
            ferr = 0.0
            for j in range(NW):
                dw = w.copy(); h = max(abs(w[j]) * 1e-6, 1e-12); dw[j] += h
                ferr += ((fwhm_from_widths(dw) - f) / h * we[j]) ** 2
            if pol == "V":
                i0 = sl.start
                cv = pcov[i0, i0] + pcov[i0 + NPJ - 1, i0 + NPJ - 1] \
                    + 2 * pcov[i0, i0 + NPJ - 1]
                cen, cen_err = pk[0] + pk[-1], np.sqrt(max(cv, 0))
            else:
                cen, cen_err = pk[0], ek[0]
            rows.append(dict(polarization=pol, center=cen, center_err=cen_err,
                             fwhm=f, fwhm_err=np.sqrt(ferr),
                             area=pk[1 + k], area_err=ek[1 + k]))
    rows.sort(key=lambda r: (r["polarization"], r["center"]))
    for pol in POLS:
        for j, r in enumerate([r for r in rows if r["polarization"] == pol], 1):
            r["peak"] = j
    return rows


# --------------------------------------------------------------- plotting
COLORS = {"H": "tab:blue", "V": "tab:red"}


def plot_pair(xidx, pr, res, outpath, sample="S"):
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    for k, pol in enumerate(POLS):
        x, y = pr.d[pol]
        c = COLORS[pol]
        name = "H (horizontal)" if pol == "H" else "V (vertical)"
        ax.plot(x, y, ".", ms=3, color=c, alpha=0.5, label=f"{name} data")
        if res is None:
            continue
        xf = np.linspace(x.min(), x.max(), 3000)
        ax.plot(xf, eval_pol(xf, res["popt"], res["n"], pol, pr.x0), "-",
                color=c, lw=1.8,
                label=f"{pol} fit (R²={r_squared(pr, res, pol):.4f})")
        bg = background(xf, res["popt"][k * NBG:(k + 1) * NBG], pr.x0)
        for i in range(res["n"]):
            sp = single_params(res["popt"][peak_slice(i)], pol)
            comp = PEAK_FN(xf, *sp) + bg
            ax.fill_between(xf, bg, comp, color=c, alpha=0.12)
            ax.plot(xf, comp, "--", color=c, lw=0.8)
        axr.plot(x, res["resid"][pol], ".", ms=2, color=c, alpha=0.6)
    if res is not None:                       # mark shared peak centres once
        for i in range(res["n"]):
            c0 = res["popt"][peak_slice(i)][0]
            ax.axvline(c0, color=COLORS["H"], lw=0.6, ls=":")
            if MAX_SHIFT > 0:
                ax.axvline(c0 + res["popt"][peak_slice(i)][-1],
                           color=COLORS["V"], lw=0.6, ls=":")
            ax.annotate(f"{c0:.1f}", (c0, 1.0), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=7, color="0.3")
        ax.text(0.01, 0.98, f"{res['n']} peaks (V shift ≤ ±{MAX_SHIFT:g})", transform=ax.transAxes,
                va="top", fontsize=8)
    axr.axhline(0, color="k", lw=0.6)
    ax.set_ylabel(Y_LABEL)
    axr.set_ylabel("Residual")
    axr.set_xlabel(X_LABEL)
    ax.set_title(f"{sample}, polarizer x = {xidx}  (rotation {ANGLE_STEP * xidx}°)",
                 pad=14)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------- main
def main():
    pairs = find_pairs(DATA_DIR)
    if not pairs:
        found = [f.name for f in DATA_DIR.iterdir() if f.suffix.lower() == ".csv"]
        print("CSV files actually in the folder:", found[:10] or "none")
        sys.exit(f"No files matching {FILE_REGEX} in {DATA_DIR.resolve()}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Found {len(pairs)} H/V pairs; shape = {PEAK_SHAPE}, "
          f"FWHM {MIN_FWHM}-{MAX_FWHM}")

    peak_out, summary = [], []
    for xidx, files in pairs.items():
        pr = Pair(*load_spectrum(files["H"]), *load_spectrum(files["V"]))
        try:
            res = fit_pair(pr, FORCE_N_PEAKS.get(xidx))
        except Exception as err:
            print(f"  ! fit failed for x={xidx}: {err}")
            res = None
        m = re.search(r"(S\d+)_", files["H"].name)
        sample = m.group(1) if m else "S"
        plot_pair(xidx, pr, res,
                  OUTPUT_DIR / f"{sample}_x{xidx:02d}_{ANGLE_STEP * xidx:03d}deg.png",
                  sample)
        if res is None:
            continue
        base = dict(x_index=xidx, rotation_deg=ANGLE_STEP * xidx)
        summary.append({**base, "n_peaks": res["n"],
                        "r2_H": r_squared(pr, res, "H"),
                        "r2_V": r_squared(pr, res, "V"), "bic": res["bic"],
                        "file_H": files["H"].name, "file_V": files["V"].name})
        for r in peak_rows(res):
            peak_out.append({**base, **r})
        cs = ", ".join(
            f"{pk[0]:.1f}({pk[-1]:+.1f})"
            for pk in sorted((res["popt"][peak_slice(i)] for i in range(res["n"])),
                             key=lambda q: q[0]))
        print(f"  x={xidx:2d} ({ANGLE_STEP * xidx:3d}°): {res['n']} peaks, H centre(V shift): {cs}")

    cols = ["x_index", "rotation_deg", "polarization", "peak", "center",
            "center_err", "fwhm", "fwhm_err", "area", "area_err"]
    peaks = pd.DataFrame(peak_out, columns=cols)
    peaks.to_csv(OUTPUT_DIR / "peak_fits.csv", index=False)
    pd.DataFrame(summary).to_csv(OUTPUT_DIR / "fit_summary.csv", index=False)
    print(f"\nSaved plots, peak_fits.csv and fit_summary.csv to "
          f"{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()