import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Expected column name fragments (case-insensitive) in the ILO Excel file
_COL_MAP = {
    "code": re.compile(r"isco.*(code|group)", re.IGNORECASE),
    "title": re.compile(r"title", re.IGNORECASE),
    "definition": re.compile(r"definition", re.IGNORECASE),
    "tasks": re.compile(r"task", re.IGNORECASE),
    "included": re.compile(r"includ", re.IGNORECASE),
}

# ISCO-08 major group labels
MAJOR_GROUP_LABELS = {
    "1": "Managers",
    "2": "Professionals",
    "3": "Technicians & assoc. professionals",
    "4": "Clerical support workers",
    "5": "Service & sales workers",
    "6": "Skilled agricultural workers",
    "7": "Craft & related trades workers",
    "8": "Plant & machine operators",
    "9": "Elementary occupations",
    "0": "Armed forces",
}


def _detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """Match DataFrame columns to canonical names using regex patterns."""
    mapping: dict[str, str] = {}
    for canonical, pattern in _COL_MAP.items():
        for col in df.columns:
            if pattern.search(str(col)):
                mapping[canonical] = col
                break
    return mapping


def _is_valid_isco_code(val) -> bool:
    """Return True if val looks like a 1–4 digit ISCO code."""
    try:
        s = str(val).strip()
        # Allow integer-like strings of length 1–4
        return bool(re.fullmatch(r"\d{1,4}", s))
    except Exception:
        return False


def parse_isco(path: str) -> pd.DataFrame:
    """Parse the ISCO-08 Excel file into a clean DataFrame.

    The ILO Excel file uses a hierarchical layout where:
    - 1-digit codes are major groups
    - 2-digit codes are sub-major groups
    - 3-digit codes are minor groups
    - 4-digit codes are unit groups

    Args:
        path: Path to the ISCO-08 Excel file.

    Returns:
        DataFrame with columns: code, title, definition, tasks, included, major_group, level.
    """
    logger.info("Parsing ISCO-08 file: %s", path)

    # Read without assuming header row — we'll detect it
    raw = pd.read_excel(path, engine="openpyxl", header=None, dtype=str)

    # Find the header row: the row that contains "title" (case-insensitive)
    header_row = None
    for i, row in raw.iterrows():
        vals = [str(v).lower() for v in row if pd.notna(v)]
        if any("title" in v for v in vals):
            header_row = i
            break

    if header_row is None:
        # Fall back: try row 0
        header_row = 0
        logger.warning("Could not detect header row; using row 0")

    df = pd.read_excel(
        path,
        engine="openpyxl",
        header=header_row,
        dtype=str,
    )

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    col_map = _detect_columns(df)
    logger.info("Detected column mapping: %s", col_map)

    missing = [k for k in ("code", "title") if k not in col_map]
    if missing:
        # Try positional fallback: first col = code, second = title
        logger.warning(
            "Could not detect columns %s by name; using positional fallback", missing
        )
        cols = list(df.columns)
        if "code" not in col_map and len(cols) >= 1:
            col_map["code"] = cols[0]
        if "title" not in col_map and len(cols) >= 2:
            col_map["title"] = cols[1]
        if "definition" not in col_map and len(cols) >= 3:
            col_map["definition"] = cols[2]
        if "tasks" not in col_map and len(cols) >= 4:
            col_map["tasks"] = cols[3]
        if "included" not in col_map and len(cols) >= 5:
            col_map["included"] = cols[4]

    # Build canonical DataFrame
    canonical_cols = ["code", "title", "definition", "tasks", "included"]
    result = pd.DataFrame()
    for canonical in canonical_cols:
        if canonical in col_map:
            result[canonical] = df[col_map[canonical]]
        else:
            result[canonical] = pd.NA

    # Keep only rows with valid ISCO codes
    result = result[result["code"].apply(_is_valid_isco_code)].copy()

    # Normalize codes: strip whitespace, remove decimal (e.g., "1.0" → "1")
    result["code"] = result["code"].str.strip().str.replace(r"\.0$", "", regex=True)

    # Add derived columns
    result["major_group"] = result["code"].str[0]
    result["level"] = result["code"].str.len().map({1: "major", 2: "sub_major", 3: "minor", 4: "unit"})

    # Clean up string columns
    for col in ["title", "definition", "tasks", "included"]:
        result[col] = result[col].where(result[col].notna(), other="")
        result[col] = result[col].str.strip()

    result = result.reset_index(drop=True)
    logger.info(
        "Parsed %d occupations (%d unit groups)",
        len(result),
        (result["level"] == "unit").sum(),
    )
    return result
