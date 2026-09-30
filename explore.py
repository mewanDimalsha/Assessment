"""Initial exploration of the Victorian Road Crash Data.

Reads every CSV in ./data and prints, per file: shape, columns, dtypes,
missing values, top-10 value_counts for categorical columns, and a set of
data quality checks. Also looks for join keys across files.

No analysis is done here - this is purely a description of the raw data.

Usage:  .venv/bin/python explore.py  [> explore_output.txt]
"""
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.data_loader import VIC_LAT_RANGE, VIC_LON_RANGE

DATA_DIR = Path(__file__).parent / "data"

# Columns with at most this many distinct values are treated as categorical,
# even if they are numeric (e.g. codes like DCA_CODE, SPEED_ZONE).
CATEGORICAL_MAX_UNIQUE = 50
TOP_N = 10

# Numeric codes commonly used in Victorian crash data for "unknown / not
# applicable / other". 9999 appears in ROAD_ROUTE_1.
SENTINEL_CODES = [777, 888, 999, 9999]
# Text labels that mean "unknown" but are not blank.
UNKNOWN_TEXT_PATTERN = r"(?i)^\s*(?:unk\.?|n/?a|\?|-)\s*$|not known|unknown|^other\b(?! injury)"

# Rough bounding box for Victoria, used to flag implausible coordinates.
VIC_LAT, VIC_LON = VIC_LAT_RANGE, VIC_LON_RANGE


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_all():
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {DATA_DIR}")
    # Read everything as its natural type, but keep empty strings as NaN.
    return {f.name: pd.read_csv(f, low_memory=False) for f in files}


def describe_basic(df):
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\nColumns and dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<28} {str(dtype):<10} unique={df[col].nunique(dropna=True):,}")

    print("\nMissing values per column (blank/NaN):")
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    table = pd.DataFrame({"missing": missing, "pct": pct})
    print(table.to_string())


def text_columns(df):
    """Text columns. pandas 3 reads strings as 'str' dtype rather than 'object',
    so check both to work on the lab's Python 3.13 environment too."""
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])]


def categorical_columns(df):
    text = set(text_columns(df))
    cols = []
    for col in df.columns:
        if col in text or df[col].nunique(dropna=True) <= CATEGORICAL_MAX_UNIQUE:
            cols.append(col)
    return cols


def describe_categoricals(df):
    print(f"\nTop {TOP_N} value_counts for categorical columns "
          f"(object dtype, or <= {CATEGORICAL_MAX_UNIQUE} distinct values):")
    for col in categorical_columns(df):
        vc = df[col].value_counts(dropna=False).head(TOP_N)
        print(f"\n-- {col} ({df[col].nunique(dropna=True):,} distinct)")
        print(vc.to_string())


def describe_numeric(df):
    num = df.select_dtypes("number")
    if num.empty:
        return
    print("\nNumeric summary:")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(num.describe().T.to_string())


def quality_checks(df):
    print("\n--- Data quality checks ---")

    # 1. Sentinel numeric codes (777/888/999/9999)
    print(f"\nSentinel codes {SENTINEL_CODES} in numeric columns:")
    found = False
    for col in df.select_dtypes("number").columns:
        hits = {c: int((df[col] == c).sum()) for c in SENTINEL_CODES}
        hits = {c: n for c, n in hits.items() if n}
        if hits:
            found = True
            print(f"  {col:<28} {hits}")
    # Sentinels also hide in text columns, e.g. "999 km/hr" or "Other speed limit"
    for col in text_columns(df):
        s = df[col].astype(str)
        hits = {c: int(s.str.contains(rf"\b{c}\b", regex=True).sum()) for c in SENTINEL_CODES}
        hits = {c: n for c, n in hits.items() if n}
        if hits:
            found = True
            print(f"  {col:<28} {hits} (text contains code)")
    if not found:
        print("  none found")

    # 2. Text values meaning 'unknown'
    print("\nText values that mean unknown/not known/other (non-blank):")
    found = False
    for col in text_columns(df):
        s = df[col].dropna().astype(str)
        mask = s.str.contains(UNKNOWN_TEXT_PATTERN, regex=True)
        if mask.any():
            found = True
            labels = s[mask].value_counts().head(5).to_dict()
            print(f"  {col:<28} {int(mask.sum()):>7,}  {labels}")
    if not found:
        print("  none found")

    # 3. Coordinates
    lat = next((c for c in df.columns if c.upper() == "LATITUDE"), None)
    lon = next((c for c in df.columns if c.upper() == "LONGITUDE"), None)
    if lat and lon:
        print("\nCoordinates:")
        missing = df[lat].isna() | df[lon].isna()
        zero = (df[lat] == 0) | (df[lon] == 0)
        outside = (~missing) & ~(
            df[lat].between(*VIC_LAT) & df[lon].between(*VIC_LON)
        )
        print(f"  missing lat/lon:            {int(missing.sum()):,}")
        print(f"  zero lat/lon:               {int(zero.sum()):,}")
        print(f"  outside Victoria bbox:      {int(outside.sum()):,}")
        if outside.any():
            print(df.loc[outside, [lat, lon]].head(5).to_string())

    # 4. Date coverage - are all years complete?
    date_col = next((c for c in df.columns if "DATE" in c.upper()), None)
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        print(f"\nDate coverage ({date_col}):")
        print(f"  unparseable dates: {int(dates.isna().sum()):,}")
        print(f"  range: {dates.min().date()} to {dates.max().date()}")
        by_year = dates.dt.year.value_counts().sort_index()
        months = dates.groupby(dates.dt.year).apply(lambda d: d.dt.month.nunique())
        first = dates.groupby(dates.dt.year).min().dt.date
        last = dates.groupby(dates.dt.year).max().dt.date
        summary = pd.DataFrame({"records": by_year, "months_present": months,
                                "first_date": first, "last_date": last})
        print(summary.to_string())
        partial = summary[summary["months_present"] < 12].index.tolist()
        print(f"  years with <12 months of data: {partial or 'none'}")
        # A sharp drop vs the median year also suggests incomplete data
        # (e.g. late-reported crashes not yet entered).
        low = summary[summary["records"] < 0.8 * summary["records"].median()].index.tolist()
        print(f"  years with <80% of median yearly records: {low or 'none'}")
        # Recent months are often under-counted because crashes are entered late.
        monthly = dates.dt.to_period("M").value_counts().sort_index()
        typical = monthly.iloc[-36:-12].median()
        print(f"  last 12 months vs median of the 24 months before ({typical:.0f}):")
        for period, n in monthly.iloc[-12:].items():
            flag = "  <-- below 80%, likely incomplete" if n < 0.8 * typical else ""
            print(f"    {period}: {n:>6,}{flag}")

    time_col = next((c for c in df.columns if "TIME" in c.upper()), None)
    if time_col:
        t = pd.to_datetime(df[time_col], format="%H:%M:%S", errors="coerce")
        print(f"\nTime ({time_col}): not in HH:MM:SS format={int(t.isna().sum()):,}, "
              f"exactly 00:00:00={int((df[time_col] == '00:00:00').sum()):,}")
        if t.isna().any():
            print(f"  examples: {df.loc[t.isna(), time_col].head(3).tolist()}")
            if lat:
                both = int((t.isna() & df[lat].isna()).sum())
                print(f"  of these, also missing coordinates: {both:,}")

    # 5. Duplicates
    print(f"\nFully duplicated rows: {int(df.duplicated().sum()):,}")

    # 6. Internal consistency of person counts (if present)
    parts = ["FATALITY", "SERIOUSINJURY", "OTHERINJURY", "NONINJURED"]
    if "TOTAL_PERSONS" in df.columns and all(p in df.columns for p in parts):
        mismatch = df[parts].sum(axis=1) != df["TOTAL_PERSONS"]
        print(f"Rows where FATALITY+SERIOUSINJURY+OTHERINJURY+NONINJURED != TOTAL_PERSONS: "
              f"{int(mismatch.sum()):,}")
    if {"MALES", "FEMALES", "TOTAL_PERSONS"} <= set(df.columns):
        diff = df["TOTAL_PERSONS"] - df["MALES"] - df["FEMALES"]
        print(f"Rows where MALES+FEMALES != TOTAL_PERSONS (sex unknown): {int((diff != 0).sum()):,}")
    if "TOTAL_PERSONS" in df.columns:
        print(f"Rows with TOTAL_PERSONS == 0: {int((df['TOTAL_PERSONS'] == 0).sum()):,}")
    if "ROAD_ROUTE_1" in df.columns:
        print(f"ROAD_ROUTE_1 == -1: {int((df['ROAD_ROUTE_1'] == -1).sum()):,}")


def candidate_keys(df):
    """Columns that uniquely identify rows in this file."""
    return [c for c in df.columns
            if df[c].notna().all() and df[c].is_unique]


def join_keys(frames):
    section("JOIN KEYS")
    for name, df in frames.items():
        keys = candidate_keys(df)
        print(f"{name}: unique, non-null columns (candidate primary keys): {keys or 'none'}")
        for c in df.columns:
            if c.upper().endswith("_NO") or c.upper().endswith("_ID"):
                dup = int(df[c].duplicated().sum())
                print(f"  {c}: {df[c].nunique():,} distinct, {dup:,} duplicated values")

    if len(frames) < 2:
        print("\nOnly one CSV in ./data, so there is nothing to join. The file is a "
              "flattened, one-row-per-crash extract (person and vehicle counts are "
              "already aggregated into columns).")
        return

    for (a, da), (b, db) in combinations(frames.items(), 2):
        shared = sorted(set(da.columns) & set(db.columns))
        print(f"\n{a} <-> {b}: shared columns = {shared or 'none'}")
        for c in shared:
            overlap = len(set(da[c].dropna()) & set(db[c].dropna()))
            print(f"  {c}: {overlap:,} overlapping values "
                  f"(unique in {a}: {da[c].is_unique}, unique in {b}: {db[c].is_unique})")


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    frames = load_all()
    print(f"Found {len(frames)} CSV file(s) in {DATA_DIR}: {list(frames)}")

    for name, df in frames.items():
        section(f"FILE: {name}")
        describe_basic(df)
        describe_numeric(df)
        describe_categoricals(df)
        quality_checks(df)

    join_keys(frames)


if __name__ == "__main__":
    main()
