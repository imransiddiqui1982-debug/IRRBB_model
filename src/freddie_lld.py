"""
freddie_lld.py
--------------
Ingest Freddie Mac Single-Family Loan-Level **sample** files and build the
loan-month table expected by ``calibrate_prepayment``.

Freddie does not allow unauthenticated bulk download. Place files under
``data/freddie/`` after registering at Clarity Data Intelligence:

  https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

Expected sample names (pipe-delimited, no header), per vintage YYYY::

  sample_orig_YYYY.txt   — origination
  sample_svcg_YYYY.txt   — monthly performance
  or sample_perf_YYYY.txt

Join FRED PMMS (MORTGAGE30US) by monthly reporting period for prevailing
mortgage rate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .pmms import load_pmms_history, pmms_for_yyyymm

# 0-based positions from origination_data_file_header.txt (pipe-delimited):
# CLASSIC FICO|FIRST PAYMENT DATE|...|ORIGINAL INTEREST RATE|...|LOAN IDENTIFIER|...
ORIG_RATE = 12         # ORIGINAL INTEREST RATE (percent, e.g. 6.125)
ORIG_SEQ = 19          # LOAN IDENTIFIER (join key to performance file)

# Performance file (standard SFLLD monthly layout; provide header if layout differs)
PERF_SEQ = 0           # LOAN IDENTIFIER / SEQUENCE
PERF_PERIOD = 1        # MONTHLY REPORTING PERIOD YYYYMM
PERF_UPB = 2           # CURRENT ACTUAL UPB
PERF_AGE = 4           # LOAN AGE
PERF_ZB = 8            # ZERO BALANCE CODE (01 = prepaid)


def _read_pipe(path: Path, usecols: list[int]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="|",
        header=None,
        usecols=usecols,
        dtype=str,
        low_memory=False,
    )


def find_freddie_sample_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Return (orig_path, perf_path) pairs under ``root`` (recursive)."""
    root = Path(root)
    pairs = []
    for orig in sorted(root.rglob("sample_orig_*.txt")):
        year = orig.stem.replace("sample_orig_", "")
        candidates = [
            orig.with_name(f"sample_svcg_{year}.txt"),
            orig.with_name(f"sample_perf_{year}.txt"),
        ]
        # also search same directory tree
        for name in (f"sample_svcg_{year}.txt", f"sample_perf_{year}.txt"):
            hits = list(root.rglob(name))
            candidates.extend(hits)
        perf = next((p for p in candidates if p.is_file()), None)
        if perf is not None:
            pairs.append((orig, perf))
    return pairs


def load_vintage_pair(orig_path: Path, perf_path: Path) -> pd.DataFrame:
    """Load one vintage into loan-month rows (before PMMS join)."""
    orig = _read_pipe(orig_path, [ORIG_SEQ, ORIG_RATE])
    orig.columns = ["loan_id", "orig_rate"]
    orig["orig_rate"] = pd.to_numeric(orig["orig_rate"], errors="coerce") / 100.0
    # Freddie sometimes stores rate already as percent 6.5 vs 0.065
    mask_pct = orig["orig_rate"] > 1.0
    orig.loc[mask_pct, "orig_rate"] = orig.loc[mask_pct, "orig_rate"] / 100.0
    orig = orig.dropna(subset=["loan_id", "orig_rate"])

    perf = _read_pipe(perf_path, [PERF_SEQ, PERF_PERIOD, PERF_UPB, PERF_AGE, PERF_ZB])
    perf.columns = ["loan_id", "period", "current_upb", "loan_age_months", "zero_balance_code"]
    perf["current_upb"] = pd.to_numeric(perf["current_upb"], errors="coerce")
    perf["loan_age_months"] = pd.to_numeric(perf["loan_age_months"], errors="coerce")
    perf["zero_balance_code"] = perf["zero_balance_code"].fillna("").astype(str).str.strip()
    perf["period"] = perf["period"].astype(str).str.replace(r"\.0$", "", regex=True).str[:6]
    perf = perf.dropna(subset=["loan_id", "period", "current_upb", "loan_age_months"])
    perf = perf[perf["current_upb"] > 0]

    merged = perf.merge(orig, on="loan_id", how="inner")
    return merged


def build_calibration_frame(
    freddie_root: str | Path,
    *,
    max_rows: int | None = 2_000_000,
    vintages: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build calibrator input: loan_id, period, orig_rate, loan_age_months,
    current_upb, zero_balance_code, prevailing_mtg_rate.
    """
    root = Path(freddie_root)
    pairs = find_freddie_sample_pairs(root)
    if vintages:
        want = set(str(v) for v in vintages)
        pairs = [p for p in pairs if any(v in p[0].name for v in want)]
    if not pairs:
        raise FileNotFoundError(
            f"No Freddie sample_orig_*/sample_svcg_* pairs under {root}. "
            "Download sample ZIPs from Freddie Clarity SFLLD and unzip into data/freddie/."
        )

    hist = load_pmms_history(refresh=False)
    chunks = []
    n = 0
    for orig_p, perf_p in pairs:
        df = load_vintage_pair(orig_p, perf_p)
        # map PMMS
        periods = df["period"].unique()
        pmms_map = {p: pmms_for_yyyymm(p, hist) for p in periods}
        df["prevailing_mtg_rate"] = df["period"].map(pmms_map)
        df = df.dropna(subset=["prevailing_mtg_rate"])
        chunks.append(df)
        n += len(df)
        if max_rows is not None and n >= max_rows:
            break
    out = pd.concat(chunks, ignore_index=True)
    if max_rows is not None and len(out) > max_rows:
        out = out.sample(max_rows, random_state=42)
    # prepaid flag convention for calibrator
    out["zero_balance_code"] = np.where(
        out["zero_balance_code"].isin(["01", "1", "01.0"]), "01", out["zero_balance_code"]
    )
    return out[
        [
            "loan_id",
            "period",
            "orig_rate",
            "loan_age_months",
            "current_upb",
            "zero_balance_code",
            "prevailing_mtg_rate",
        ]
    ]
