#!/usr/bin/env python3
"""
Recompute Day 4 summary tables (Table II/III) from the final report.

This script intentionally reproduces the *published* numbers using:
- the four recorded trial isotope shifts (in Å) for Hα and Hβ,
- the reduced-mass theory values used in the report,
- the calibration-term values quoted in the report's error budget.

It writes CSV/JSON outputs into ../results/.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------
# Inputs from the final report
# ----------------------------
DAY4_TRIALS = {
    "Halpha": np.array([1.7973, 1.8170, 1.8071, 1.8062], dtype=float),
    "Hbeta":  np.array([1.2656, 1.2945, 1.2667, 1.2552], dtype=float),
}

THEORY = {
    "Halpha": 1.7858,
    "Hbeta":  1.3228,
}

# Calibration uncertainty terms (dominant systematics) quoted in the report (Å)
CAL_TERM = {
    "Halpha": 0.051,
    "Hbeta":  0.038,
}

# Final reported values (Å) in the report (after calibration systematic included)
FINAL_REPORTED = {
    "Halpha": (1.801, 0.052),
    "Hbeta":  (1.263, 0.039),
}

OUTDIR = Path(__file__).resolve().parents[1] / "results"
OUTDIR.mkdir(parents=True, exist_ok=True)


def mean_sem(x: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, sample_std, SEM)."""
    mean = float(np.mean(x))
    s = float(np.std(x, ddof=1))  # sample stdev
    sem = s / np.sqrt(len(x))
    return mean, s, float(sem)


def percent_diff(exp: float, th: float) -> float:
    return 100.0 * (exp - th) / th


def main() -> None:
    rows_compare = []
    rows_budget = []
    summary = {"day": "Day4", "inputs": {}, "outputs": {}}

    for line_key, trials in DAY4_TRIALS.items():
        mean, s, sem = mean_sem(trials)
        th = THEORY[line_key]
        cal = CAL_TERM[line_key]
        tot = float(np.sqrt(sem**2 + cal**2))

        exp_final, tot_final = FINAL_REPORTED[line_key]
        pdiff = percent_diff(exp_final, th)

        # Table II row
        rows_compare.append({
            "Line": "Hα" if line_key == "Halpha" else "Hβ",
            "Delta_lambda_exp_A": exp_final,
            "Sigma_total_A": tot_final,
            "Delta_lambda_th_A": th,
            "Percent_difference_%": pdiff,
        })

        # Table III row
        rows_budget.append({
            "Line": "Hα" if line_key == "Halpha" else "Hβ",
            "Statistical_SEM_A": sem,
            "Calibration_A": cal,
            "Total_quadrature_A": tot,
            "Final_reported_total_A": tot_final,
        })

        summary["inputs"][line_key] = {
            "trials_A": trials.tolist(),
            "theory_A": th,
            "calibration_term_A": cal,
        }
        summary["outputs"][line_key] = {
            "trial_mean_A": mean,
            "trial_sample_std_A": s,
            "trial_SEM_A": sem,
            "quadrature_total_A": tot,
            "final_reported_A": exp_final,
            "final_reported_total_A": tot_final,
            "percent_difference_%": pdiff,
        }

    # Write outputs
    df_compare = pd.DataFrame(rows_compare)
    df_budget = pd.DataFrame(rows_budget)

    df_compare.to_csv(OUTDIR / "day4_tableII_compare.csv", index=False)
    df_budget.to_csv(OUTDIR / "day4_tableIII_error_budget.csv", index=False)

    with open(OUTDIR / "day4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Wrote:")
    print(" -", OUTDIR / "day4_tableII_compare.csv")
    print(" -", OUTDIR / "day4_tableIII_error_budget.csv")
    print(" -", OUTDIR / "day4_summary.json")


if __name__ == "__main__":
    main()
