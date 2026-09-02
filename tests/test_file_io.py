"""Tests for file_io helpers."""
import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.file_io import is_excel_content, read_binary, read_csv_robust, read_excel_sheets  # noqa: E402


def test_is_excel_content_detects_xlsx():
    xlsx = os.path.join(
        os.path.dirname(__file__), "..", "data", "us_lcr_nsfr_deposit_model.xlsx"
    )
    if not os.path.exists(xlsx):
        pytest.skip("sample workbook missing")
    with open(xlsx, "rb") as f:
        data = f.read()
    assert is_excel_content(data)


def test_read_csv_rejects_xlsx():
    xlsx = os.path.join(
        os.path.dirname(__file__), "..", "data", "us_lcr_nsfr_deposit_model.xlsx"
    )
    if not os.path.exists(xlsx):
        pytest.skip("sample workbook missing")
    with pytest.raises(ValueError, match="Excel workbook"):
        read_csv_robust(xlsx)


def test_read_excel_sheets():
    xlsx = os.path.join(
        os.path.dirname(__file__), "..", "data", "us_lcr_nsfr_deposit_model.xlsx"
    )
    if not os.path.exists(xlsx):
        pytest.skip("sample workbook missing")
    sheets = read_excel_sheets(xlsx)
    assert "Customer Deposits" in sheets


def test_read_csv_utf8():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "balance_sheet_template.csv")
    df = read_csv_robust(csv_path)
    assert not df.empty


def test_read_binary_bytesio():
    payload = b"PK\x03\x04test"
    assert read_binary(io.BytesIO(payload)) == payload
