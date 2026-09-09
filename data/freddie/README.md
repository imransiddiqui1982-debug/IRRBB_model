# Freddie Mac Single-Family Loan-Level Dataset (samples)

Freddie requires a free Clarity account — files cannot be fetched automatically.

1. Open https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
2. Sign in to Clarity → SFLLD Data Download → download **sample** ZIPs by vintage
3. Unzip here so this folder contains pairs such as:

   sample_orig_2019.txt
   sample_svcg_2019.txt
   sample_orig_2020.txt
   sample_svcg_2020.txt

   (some releases use `sample_perf_YYYY.txt` instead of `sample_svcg_YYYY.txt`)

4. From the repo root:

   python scripts/calibrate_from_freddie_pmms.py

That joins FRED PMMS (MORTGAGE30US) by month and writes
`data/calibrated_prepayment_params.json` for Step B.

Step A PMMS anchoring works **without** these files (live FRED only).
