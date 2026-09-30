"""Load, merge and clean the Victorian Road Crash Data.

The data folder may contain:
  - a crash table: one row per ACCIDENT_NO (always required)
  - a person table: one row per person involved (has PERSON_ID)  - optional
  - a vehicle table: one row per vehicle involved (has VEHICLE_ID) - optional

Person and vehicle tables have many rows per crash, so they are aggregated to
one row per ACCIDENT_NO *before* merging. Otherwise a crash with 3 people would
appear 3 times and every crash-level count would be inflated.

The current extract (victorian_road_crash_data.csv) is a single crash table
with person/vehicle counts already included as columns, so no merge happens.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

KEY = "ACCIDENT_NO"

# Crashes from this date onwards are under-counted (late data entry): monthly
# totals drop well below normal from Sep 2025. See explore_output.txt.
INCOMPLETE_FROM = pd.Timestamp("2025-09-01")

# Text labels that mean the value was not recorded.
UNKNOWN_LABELS = ["Not known", "Unk.", "Unknown"]
# Column-specific "unknown" labels.
UNKNOWN_BY_COLUMN = {"LIGHT_CONDITION": ["Dark Street lights unknown"]}
# Numeric codes meaning "no route number" / invalid.
ROUTE_SENTINELS = [9999, -1]

# Rough bounding box for Victoria (degrees). Points outside it can't be in
# Victoria, so they are treated as invalid coordinates.
VIC_LAT_RANGE = (-39.3, -33.9)
VIC_LON_RANGE = (140.9, 150.1)

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------- loading ---

def find_tables(data_dir=DATA_DIR):
    """Read every CSV in data_dir and classify it as crash / person / vehicle
    from its columns. Returns a dict {"crash": df, "person": df, "vehicle": df}
    with only the tables that were found."""
    tables = {}
    for path in sorted(Path(data_dir).glob("*.csv")):
        df = pd.read_csv(path, low_memory=False)
        cols = set(df.columns)
        if KEY not in cols:
            print(f"  [data_loader] ignoring {path.name}: no {KEY} column")
            continue
        if "PERSON_ID" in cols:
            kind = "person"
        elif "VEHICLE_ID" in cols:
            kind = "vehicle"
        elif "ACCIDENT_DATE" in cols and df[KEY].is_unique:
            kind = "crash"
        else:
            print(f"  [data_loader] ignoring {path.name}: unrecognised table")
            continue
        if kind in tables:
            raise ValueError(f"More than one {kind} table found in {data_dir}")
        print(f"  [data_loader] {path.name}: {kind} table, {len(df):,} rows")
        tables[kind] = df

    if "crash" not in tables:
        raise FileNotFoundError(f"No crash-level CSV found in {data_dir}")
    return tables


def _aggregate(df, prefix, count_name, category_cols):
    """One row per ACCIDENT_NO: a row count plus counts of each category."""
    out = df.groupby(KEY).size().rename(f"{prefix}{count_name}").to_frame()
    for col in category_cols:
        if col not in df.columns:
            continue
        counts = pd.crosstab(df[KEY], df[col].fillna("missing"))
        counts.columns = [f"{prefix}{col}_{c}" for c in counts.columns]
        out = out.join(counts)
    return out.fillna(0).astype(int).reset_index()


def aggregate_persons(persons):
    """Person table -> one row per crash (P_ prefixed columns)."""
    return _aggregate(persons, "P_", "N_PERSONS",
                      ["INJ_LEVEL", "SEX", "ROAD_USER_TYPE", "AGE_GROUP"])


def aggregate_vehicles(vehicles):
    """Vehicle table -> one row per crash (V_ prefixed columns)."""
    return _aggregate(vehicles, "V_", "N_VEHICLES",
                      ["VEHICLE_TYPE", "VEHICLE_BODY_STYLE"])


def merge_tables(crash, persons=None, vehicles=None):
    """Left-merge aggregated person/vehicle tables onto the crash table.
    Raises if the merge would change the number of crashes."""
    merged = crash
    for extra in (persons, vehicles):
        if extra is None:
            continue
        merged = merged.merge(extra, on=KEY, how="left", validate="one_to_one")
        # Crashes with no matching person/vehicle rows have zero of each count.
        added = [c for c in extra.columns if c != KEY]
        merged[added] = merged[added].fillna(0).astype(int)
    if len(merged) != len(crash):
        raise AssertionError(
            f"Merge changed row count: {len(crash):,} -> {len(merged):,}")
    return merged


# --------------------------------------------------------------- cleaning ---

def _is_text(series):
    # pandas 3 uses a 'str' dtype instead of 'object' for text columns.
    return not pd.api.types.is_numeric_dtype(series) and \
        not pd.api.types.is_datetime64_any_dtype(series)


def clean(df):
    """Return a cleaned copy: unknown codes -> NaN, parsed dates/times,
    numeric speed zone and derived time columns."""
    df = df.copy()

    # Unknown labels -> NaN
    for col in df.columns:
        if _is_text(df[col]):
            labels = UNKNOWN_LABELS + UNKNOWN_BY_COLUMN.get(col, [])
            df[col] = df[col].where(~df[col].isin(labels))
    if "ROAD_ROUTE_1" in df.columns:
        df["ROAD_ROUTE_1"] = df["ROAD_ROUTE_1"].where(
            ~df["ROAD_ROUTE_1"].isin(ROUTE_SENTINELS))

    # Speed zone "60 km/hr" / "30km/hr" -> 60.0 / 30.0; other labels -> NaN
    if "SPEED_ZONE" in df.columns:
        df["SPEED_ZONE_KMH"] = pd.to_numeric(
            df["SPEED_ZONE"].str.extract(r"^(\d+)\s*km/hr$", expand=False),
            errors="coerce")

    # Dates and times. 85 rows store times as "HH:MM:SS.0000000".
    df["ACCIDENT_DATE"] = pd.to_datetime(df["ACCIDENT_DATE"], errors="coerce")
    if "ACCIDENT_TIME" in df.columns:
        time_str = df["ACCIDENT_TIME"].astype(str).str.slice(0, 8)
        times = pd.to_datetime(time_str, format="%H:%M:%S", errors="coerce")
        df["HOUR"] = times.dt.hour
    df["YEAR"] = df["ACCIDENT_DATE"].dt.year
    df["MONTH"] = df["ACCIDENT_DATE"].dt.month
    if "DAY_OF_WEEK" in df.columns:
        df["DAY_OF_WEEK"] = pd.Categorical(df["DAY_OF_WEEK"],
                                           categories=DAY_ORDER, ordered=True)

    df["INCOMPLETE_PERIOD"] = df["ACCIDENT_DATE"] >= INCOMPLETE_FROM
    return df


def load_data(clean_data=True, data_dir=DATA_DIR):
    """Load all tables, aggregate person/vehicle tables, merge on ACCIDENT_NO
    and (optionally) clean. This is the entry point investigations use."""
    tables = find_tables(data_dir)
    persons = aggregate_persons(tables["person"]) if "person" in tables else None
    vehicles = aggregate_vehicles(tables["vehicle"]) if "vehicle" in tables else None
    df = merge_tables(tables["crash"], persons, vehicles)
    return clean(df) if clean_data else df


def load_raw(data_dir=DATA_DIR):
    """Merged but uncleaned data - used to report data quality issues."""
    return load_data(clean_data=False, data_dir=data_dir)


__all__ = ["load_data", "load_raw", "clean", "find_tables", "merge_tables",
           "aggregate_persons", "aggregate_vehicles", "INCOMPLETE_FROM",
           "DATA_DIR"]
