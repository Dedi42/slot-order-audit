# Reproduce all public results of the paper (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1
# Set $py to your Python, e.g. "C:/Users/<you>/anaconda3/envs/<env>/python.exe"
$py = "python"
Set-Location $PSScriptRoot

# Section IV-B/C: audit of WM-811K and shuffled-slot control (skip if already done)
if (-not (Test-Path "out/audit_real/lot_slot_stats.csv"))    { & $py s1_wm811k_audit.py --perm 2000 --out out/audit_real }
if (-not (Test-Path "out/audit_control/lot_slot_stats.csv")) { & $py s1_wm811k_audit.py --perm 2000 --out out/audit_control --shuffle-control }
& $py s2_wm811k_dataset_tests.py                 # Table III, strata, Fig. 3, analytic and fixed-threshold shares
& $py s3_wm811k_minp_rerun.py --perm 100000      # lots at p = 1/(B+1), BH after re-run ("After BH" in Table III)
& $py s4_wm811k_spikein.py --n-lots 2000         # recall under known effects

# Section IV-C: null data sets (skipped if the data are missing)
if (Test-Path "data/Wafer") { & $py s5_kai_null.py --dir data/Wafer --max-tests 4 --slot-map first25 }
if (Test-Path "data/phm_data_challenge_2018") { & $py s6_phm2018_drift.py --dir data/phm_data_challenge_2018 }

# Section VI: sampling-plan simulation, Table VII and Fig. 7
& $py s7_sampling_power.py --R 400

Write-Host "`nDone. Results are in out/"
