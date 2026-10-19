# slot-order-audit

# Slot-order audit

Code for the paper **"Outcome-Agnostic Slot-Order Auditing of Semiconductor Lots: Chamber, Start-of-Lot, and Drift Signatures in Public Wafer Maps and Two Production Lines"**
by Dedi Surahman and Yu-Ming Hsieh.

The audit tests whether the order in which the wafers of a lot deviate (odd/even, start-of-lot,
trend, period-k) is a genuine signal: within-lot permutation tests, a pooled variant for sparsely
measured lots, a shuffled-slot negative control, dataset-level inference, a mixed-effects model that
separates slot from chamber, and a simulation for metrology sampling plans.

## Installation

```bash
pip install -r requirements.txt
```
Python 3.9 or newer. The public data sets are **not** redistributed; see `data/README.md`.

## Reproducing the paper

| Script | Paper | What it produces |
|---|---|---|
| `slot_audit.py` | Sec. III-A, Eqs. (2)-(4) | core functions: slot statistics, permutation test, period-k test, pooled sparse-lot test, BH, Storey |
| `s1_wm811k_audit.py` | Sec. IV-B/C | per-lot statistics and p-values on WM-811K; `--shuffle-control` gives the negative control |
| `s2_wm811k_dataset_tests.py` | Table III, Fig. 3, Sec. IV-C | shares, binomial tests, Storey, BH, strata, analytic and fixed-threshold shares |
| `s3_wm811k_minp_rerun.py` | Sec. IV-B, Table III "After BH" | re-runs lots at p = 1/(B+1) with B = 10^5 |
| `s4_wm811k_spikein.py` | Sec. IV-C | recall under injected effects (0.25 / 0.5 / 1.0 SD) |
| `s5_kai_null.py` | Sec. IV-C | null data set: Infineon/KAI simulated lots |
| `s6_phm2018_drift.py` | Sec. IV-C | null data set: PHM 2018 sensor drift along run order |
| `s7_sampling_power.py` | Sec. VI, Table VII, Fig. 7 | minimum detectable effects and lots for 80 % power |
| `s8_fab_slot_chamber.py` | Sec. V-A to V-C, Tables IV-V | the production analysis, to run on **your own** data |
| `s9_auroc_bootstrap.py` | Sec. V-D, Table VI | AUROC with stratified bootstrap CIs |

Run everything that uses public data:

```bash
./run_all.sh                                          # Linux / macOS
powershell -ExecutionPolicy Bypass -File run_all.ps1  # Windows
```
All results are written to `out/`. Seeds are fixed in every script.
The complete WM-811K audit (26,741 lots, B = 2,000) takes about 10 min on a 16-core workstation.

## Using the audit on your own data

`s8_fab_slot_chamber.py` expects one row per wafer with the columns
`lot, slot, chamber, tool, time, y` (optional `pair`, `pmcount`); rows without a measurement keep
`y` empty. A synthetic example:

```bash
python examples/make_synthetic_fab.py --out data/synthetic_fab.csv
python s8_fab_slot_chamber.py --csv data/synthetic_fab.csv --boot 200
```

For any other outcome (wafer-map fail fraction, metrology value, robust z-score of an FDC feature,
0/1 label), call the functions of `slot_audit.py` directly:

```python
import numpy as np
from slot_audit import lot_stats, pooled_test
rng = np.random.default_rng(0)
(d_par, d_head, rho), (p_par, p_head, p_rho) = lot_stats(slots, y, B=2000, rng=rng)  # >= 20 measured slots
pooled = pooled_test([(slots_l, y_l) for ...], "head", B=2000, rng=rng)            # sparse lots
```

## Data availability

The production data of Fab A and Plant B cannot be shared under the data agreements with the two
plants. All public results can be reproduced with the scripts above.

## Citation

Please cite the paper if you use this code (reference will be added after publication).

## License

MIT, see `LICENSE`.
