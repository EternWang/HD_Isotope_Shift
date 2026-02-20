# Analysis scripts

## 1) Recompute the Day 4 tables (robust)
This reproduces Table II/III from the final report using the recorded Day 4 trial shifts.

```bash
python recompute_tables_day4.py
```

Outputs are written to `../results/`.

## 2) Optional: refit raw CSV scans
Raw scan CSVs are 2-column files:

- column 1: time (s)
- column 2: detector signal (V)

Because peak overlap and baseline drift can make nonlinear fitting sensitive to window selection,
`fit_from_raw.py` supports **per-file windows** stored in `windows.json`.

Batch example:

```bash
python fit_from_raw.py --batch ../data/hd/day4_new_lamp --windows windows.json --plot
```

This will print fitted peak centers and save overlay plots into `../results/`.
