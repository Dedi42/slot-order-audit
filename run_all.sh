#!/usr/bin/env bash
# Reproduce all public results of the paper (Linux / macOS).
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
[ -f out/audit_real/lot_slot_stats.csv ]    || $PY s1_wm811k_audit.py --perm 2000 --out out/audit_real
[ -f out/audit_control/lot_slot_stats.csv ] || $PY s1_wm811k_audit.py --perm 2000 --out out/audit_control --shuffle-control
$PY s2_wm811k_dataset_tests.py
$PY s3_wm811k_minp_rerun.py --perm 100000
$PY s4_wm811k_spikein.py --n-lots 2000
if [ -d data/Wafer ]; then $PY s5_kai_null.py --dir data/Wafer --max-tests 4 --slot-map first25; fi
if [ -d data/phm_data_challenge_2018 ]; then $PY s6_phm2018_drift.py --dir data/phm_data_challenge_2018; fi
$PY s7_sampling_power.py --R 400
echo "Done. Results are in out/"
