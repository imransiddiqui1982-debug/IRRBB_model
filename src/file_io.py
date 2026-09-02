"""
file_io.py
----------
Robust binary / CSV / Excel loading helpers.

Prevents UTF-8 decode errors when Excel workbooks are mistaken for text CSV.
"""

from __future__ import annotations

import io
import os
from typing import BinaryIO, Union

import pandas as pd

FileSource = Union[str, bytes, bytearray, BinaryIO, io.BytesIO]


def read_binary(source: FileSource) -> bytes:
    """Read raw bytes from a path, bytes-like, or file object."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, io.StringIO):
        return source.getvalue().encode("utf-8")
    if isinstance(source, str):
        if os.path.isfile(source):
            with open(source, "rb") as f:
                return f.read()
        return source.encode("utf-8")
    if hasattr(source, "getvalue"):
        val = source.getvalue()
        if isinstance(val, str):
            return val.encode("utf-8")
        return val
    if hasattr(source, "read"):
        pos = source.tell() if hasattr(source, "tell") else None
        raw = source.read()
        if pos is not None and hasattr(source, "seek"):
            source.seek(pos)
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, str):
            return raw.encode("utf-8")
        return bytes(raw)
    raise TypeError(f"Unsupported source type: {type(source)!r}")


def is_excel_content(data: bytes) -> bool:
    """True for xlsx/xls zip or OLE signatures."""
    if len(data) < 4:
        return False
    if data[:2] == b"PK":
        return True
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return True
    return False


def read_csv_robust(source: FileSource) -> pd.DataFrame:
    """Load CSV with UTF-8 / BOM / Windows-Japanese / Latin-1 fallbacks."""
    data = read_binary(source)
    if is_excel_content(data):
        raise ValueError(
            "The uploaded file looks like an Excel workbook (.xlsx), not a CSV. "
            "Upload it under 'NMD customer deposit file', or use the balance sheet CSV template."
        )
    last_err: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp932", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
    if last_err is not None:
        raise last_err
    return pd.read_csv(io.BytesIO(data), encoding="latin-1")


def read_excel_sheets(source: FileSource) -> dict[str, pd.DataFrame]:
    """Load all sheets from an Excel workbook via binary-safe read."""
    data = read_binary(source)
    if not is_excel_content(data):
        raise ValueError(
            "Expected an Excel workbook (.xlsx). "
            "For balance sheet data, upload a CSV file instead."
        )
    xl = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    return {sheet: pd.read_excel(xl, sheet_name=sheet) for sheet in xl.sheet_names}
