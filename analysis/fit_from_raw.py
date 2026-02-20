#!/usr/bin/env python3
"""
Fit a two-Gaussian + linear baseline model to a scan segment and extract Δt.

This is an *optional* helper script for refitting raw CSV scans.
Because peak overlap and baseline drift can make the fit sensitive to the chosen time window,
we recommend defining per-file windows in windows.json.

Usage examples:
  python fit_from_raw.py --csv ../data/hd/day4_new_lamp/6559-6561_L-H_trail_2.csv --expected-dt 21.6
  python fit_from_raw.py --batch ../data/hd/day4_new_lamp --windows windows.json

Outputs:
  - prints fitted peak centers and Δt
  - writes an overlay plot into ../results/ (if --plot is used)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=",")
    return arr[:, 0], arr[:, 1]


def model(t, A1, mu1, sig1, A2, mu2, sig2, c0, c1):
    return (
        A1 * np.exp(-(t - mu1) ** 2 / (2 * sig1**2))
        + A2 * np.exp(-(t - mu2) ** 2 / (2 * sig2**2))
        + c0
        + c1 * t
    )


def fit_window(t, y, left, right, mu1_guess, mu2_guess):
    m = (t >= left) & (t <= right)
    tw = t[m]
    yw = y[m]

    # downsample for speed
    n = len(tw)
    ds = max(1, n // 30000)
    tf = tw[::ds]
    yf = yw[::ds]

    base = np.percentile(yf, 5)
    A_guess = max(1e-6, np.max(yf) - base)
    sig_guess = max(0.05, (right - left) / 20)

    p0 = [A_guess, mu1_guess, sig_guess, A_guess / 2, mu2_guess, sig_guess, base, 0.0]
    bounds_lo = [0, left, 1e-4, 0, left, 1e-4, -np.inf, -np.inf]
    bounds_hi = [np.inf, right, (right-left), np.inf, right, (right-left), np.inf, np.inf]

    popt, pcov = curve_fit(model, tf, yf, p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=200000)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, (tf, yf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, help="Path to a single CSV file")
    ap.add_argument("--expected-dt", type=float, default=None, help="Expected Δt (s) for peak selection")
    ap.add_argument("--left", type=float, default=None)
    ap.add_argument("--right", type=float, default=None)
    ap.add_argument("--plot", action="store_true", help="Save overlay plot to ../results/")
    ap.add_argument("--batch", type=str, help="Batch folder of CSV files (uses --windows)")
    ap.add_argument("--windows", type=str, default="windows.json", help="JSON file with per-file windows")
    args = ap.parse_args()

    outdir = Path(__file__).resolve().parents[1] / "results"
    outdir.mkdir(exist_ok=True, parents=True)

    def run_one(csv_path: Path, expected_dt: float | None, left: float | None, right: float | None):
        t, y = load_csv(csv_path)

        # default window: use provided or auto from signal region
        if left is None or right is None:
            # use region where signal exceeds a threshold
            thr = np.percentile(y, 95) * 0.2
            m = y > thr
            if np.any(m):
                left = float(max(t[0], t[m].min() - 10))
                right = float(min(t[-1], t[m].max() + 10))
            else:
                left, right = float(t[0]), float(t[-1])

        # choose initial mu guesses from smoothed curve
        tw = t[(t >= left) & (t <= right)]
        yw = y[(t >= left) & (t <= right)]
        ys = gaussian_filter1d(yw, sigma=200)
        # candidate peaks: take top few local maxima
        idx = np.argsort(ys)[-8:]
        cand = np.sort(tw[idx])

        # pick two candidates with separation closest to expected_dt if provided
        if expected_dt is not None and len(cand) >= 2:
            best = None
            best_score = 1e99
            for i in range(len(cand)):
                for j in range(i + 1, len(cand)):
                    dt = abs(cand[j] - cand[i])
                    score = ((dt - expected_dt) / expected_dt) ** 2
                    if score < best_score:
                        best_score = score
                        best = (cand[i], cand[j])
            mu1_guess, mu2_guess = best
        else:
            # fallback: use two highest points
            mu1_guess, mu2_guess = cand[-2], cand[-1]

        popt, perr, (tf, yf) = fit_window(t, y, left, right, mu1_guess, mu2_guess)
        A1, mu1, sig1, A2, mu2, sig2, c0, c1 = popt
        dt = abs(mu2 - mu1)

        print(f"\n{csv_path.name}")
        print(f"  window: [{left:.3f}, {right:.3f}] s")
        print(f"  mu1 = {mu1:.6f} s, mu2 = {mu2:.6f} s, Δt = {dt:.6f} s")

        if args.plot:
            tt = np.linspace(left, right, 2000)
            plt.figure()
            plt.plot(tf, yf, label="Data (downsampled)")
            plt.plot(tt, model(tt, *popt), label="Fit")
            plt.xlabel("Time (s)")
            plt.ylabel("Signal (V)")
            plt.legend()
            plt.tight_layout()
            outpath = outdir / f"fit_{csv_path.stem}.png"
            plt.savefig(outpath, dpi=200)
            plt.close()
            print(f"  saved plot: {outpath}")

    if args.batch:
        w = json.loads(Path(args.windows).read_text(encoding="utf-8"))
        batch_dir = Path(args.batch)
        for csv_path in sorted(batch_dir.glob("*.csv")):
            cfg = w.get(csv_path.name, {})
            run_one(
                csv_path,
                expected_dt=cfg.get("expected_dt_s", args.expected_dt),
                left=cfg.get("left_s", args.left),
                right=cfg.get("right_s", args.right),
            )
    else:
        if not args.csv:
            raise SystemExit("Provide --csv or --batch.")
        run_one(Path(args.csv), expected_dt=args.expected_dt, left=args.left, right=args.right)


if __name__ == "__main__":
    main()
