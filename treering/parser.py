"""Parser for Tucson-format .rwl (ring-width) files.

Tucson Decadal Format
---------------------
A standard .rwl file may begin with up to three header lines whose first field
ends with a single digit 1, 2, or 3 (e.g. ``JUPRO1 1 ...``).  These header
lines carry site metadata and are **not** measurement data.

Data lines have the structure::

    <series_id>  <decade_start_year>  <value_1> ... <value_N>

* **series_id** — up to 8 characters, left-justified.
* **decade_start_year** — the calendar year of the first measurement on that
  line.  Subsequent values are for consecutive years.
* Each value is an integer ring-width measurement (units depend on the
  dataset; often 1/100 mm or 1/1000 mm).
* A value of ``999`` signals the end of a series (stop marker).
* A value of ``-9999`` is sometimes used as a missing-value marker.

Multiple series are stored sequentially in the same file.

References
----------
* ITRDB / NOAA Paleoclimatology Tucson format specification.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Tucson stop-marker value.  A measurement equal to this terminates a series.
_STOP_MARKER = 999

# Header-line pattern: ``<ID> <1|2|3> <rest>``
# The ID for headers typically includes a digit suffix, then a space, then
# the header-sequence number (1, 2, or 3).
_HEADER_RE = re.compile(
    r"^(\S+)\s+([123])\s+",
)

# Data-line pattern: <series_id> <year> <tokens ...>
# series_id: 1-8 non-whitespace chars
# year:      integer (possibly negative for BC dates)
# tokens:    whitespace-separated values (validated individually)
_DATA_RE = re.compile(
    r"^(\S{1,8})\s+(-?\d+)((?:\s+\S+)+)\s*$",
)


class RWLParseError(Exception):
    """Raised when a Tucson .rwl record is malformed."""


def parse_rwl(
    filepath: Union[str, Path],
) -> pd.DataFrame:
    """Parse a Tucson-format ``.rwl`` file into a tidy DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.rwl`` file.

    Returns
    -------
    pd.DataFrame
        Columns: ``series_id`` (str), ``year`` (int), ``ring_width`` (int).
        Sorted by ``(series_id, year)``.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    RWLParseError
        If the file is empty, contains no valid data lines, or has
        malformed records.
    ValueError
        If non-numeric measurement values are encountered.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    if not lines or all(line.strip() == "" for line in lines):
        raise RWLParseError(f"Input file is empty: {path}")

    records: list[dict[str, object]] = []
    header_ids_seen: set[str] = set()
    line_num = 0

    for raw_line in lines:
        line_num += 1
        line = raw_line.strip()
        if not line:
            continue

        # --- Detect header lines ------------------------------------------------
        # Header lines match ``<ID> <1|2|3> <text>`` where <1|2|3> is a single
        # digit acting as a header-sequence number (not a year).
        hdr_match = _HEADER_RE.match(line)
        if hdr_match:
            hdr_id = hdr_match.group(1)
            hdr_seq = hdr_match.group(2)
            # Track header IDs so we can distinguish them from data lines
            # whose second token happens to be 1, 2, or 3 (a valid year).
            # A true header line has a *non-numeric* remainder after the
            # sequence digit (site names, species, etc.) and typically has
            # the same base ID across all three lines.
            remainder = line[hdr_match.end():].strip()
            # If the remainder starts with many integers that look like
            # ring-width data, treat this as a data line instead.
            remainder_tokens = remainder.split()
            if remainder_tokens and not all(
                _is_integer_token(t) for t in remainder_tokens
            ):
                header_ids_seen.add(hdr_id)
                logger.debug("Skipping header line %d: %s", line_num, line[:60])
                continue

        # --- Parse data line ----------------------------------------------------
        data_match = _DATA_RE.match(line)
        if not data_match:
            # Line doesn't match data pattern — skip comment / blank
            logger.debug(
                "Skipping non-data line %d: %s", line_num, line[:60]
            )
            continue

        series_id = data_match.group(1)
        year_str = data_match.group(2)
        values_str = data_match.group(3).strip()

        # Skip header IDs that leaked through the pattern
        if series_id in header_ids_seen:
            logger.debug(
                "Skipping header-continuation line %d: %s",
                line_num,
                line[:60],
            )
            continue

        try:
            start_year = int(year_str)
        except ValueError as exc:
            raise RWLParseError(
                f"Line {line_num}: non-integer year '{year_str}' in "
                f"series '{series_id}'"
            ) from exc

        raw_values = values_str.split()
        for idx, token in enumerate(raw_values):
            try:
                value = int(token)
            except ValueError as exc:
                raise RWLParseError(
                    f"Line {line_num}: non-numeric measurement '{token}' "
                    f"at position {idx} in series '{series_id}'"
                ) from exc

            if value == _STOP_MARKER:
                # End-of-series marker — do not record it as a measurement.
                break

            year = start_year + idx

            if not np.isfinite(float(value)):
                raise RWLParseError(
                    f"Line {line_num}: non-finite measurement {value} "
                    f"for year {year} in series '{series_id}'"
                )

            records.append(
                {
                    "series_id": series_id,
                    "year": year,
                    "ring_width": value,
                }
            )

    if not records:
        raise RWLParseError(
            f"No valid measurement records found in {path}"
        )

    df = pd.DataFrame(records)
    df["year"] = df["year"].astype(int)
    df["ring_width"] = df["ring_width"].astype(int)
    df = df.sort_values(["series_id", "year"]).reset_index(drop=True)

    # Validate: no duplicate (series_id, year) pairs
    dupes = df.duplicated(subset=["series_id", "year"], keep=False)
    if dupes.any():
        first_dupe = df.loc[dupes].iloc[0]
        logger.warning(
            "Duplicate (series_id, year) detected: series=%s, year=%d. "
            "Keeping first occurrence.",
            first_dupe["series_id"],
            first_dupe["year"],
        )
        df = df.drop_duplicates(subset=["series_id", "year"], keep="first")
        df = df.reset_index(drop=True)

    n_series = df["series_id"].nunique()
    n_obs = len(df)
    logger.info(
        "Parsed %d observations across %d series from %s",
        n_obs,
        n_series,
        path.name,
    )

    return df


def _is_integer_token(token: str) -> bool:
    """Return True if *token* can be parsed as an integer."""
    try:
        int(token)
        return True
    except ValueError:
        return False
