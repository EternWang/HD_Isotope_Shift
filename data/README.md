# Data

All raw scans are stored as 2-column CSV files:

1. **time** (seconds)  
2. **detector signal** (volts)

## Folder meaning
- `hg/day1/`, `hg/day2/`  
  Mercury lamp scans used to calibrate the scan-rate \(\beta = d\lambda/dt\).
- `hd/day3_old_lamp/`  
  Mixed H–D lamp (old lamp). Lower SNR / more ambiguous peak heights. Used as a control dataset.
- `hd/day4_new_lamp/`  
  Mixed H–D lamp (new lamp). Four repeated scans around Hα and Hβ used for final results.

## Metadata
`metadata.csv` provides one row per file and includes:
- day label,
- lamp type,
- spectral region (Hα / Hβ / etc.),
- nominal sweep rate (if known from notes),
- scan direction (if encoded in filename).

If you rename files, update `metadata.csv` accordingly.
