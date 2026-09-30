"""Run all road safety investigations in order.

Usage:  python main.py
Figures are saved to ./figures as 300 dpi PNGs, tables to ./results as CSV.

To add an investigation: create a module with a run(df) function that returns
the list of saved file paths, then add it to INVESTIGATIONS below.
  df  - cleaned data (unknown codes -> NaN, parsed dates, SPEED_ZONE_KMH, ...)
Investigations that report on the uncleaned data take run(df, raw) instead and
are listed with needs_raw=True.
"""
import utils.transport
import utils.session
import time

from src.data_loader import clean, load_raw
from src import (inv1_speed_zone, inv2_time_of_day, inv3_yearly_trend,
                 inv4_light_condition, inv5_hotspots, inv6_severity_model,
                 deep_investigation)
from src.investigations import data_quality
from src.utils import set_style

# (name, run function, needs_raw), run in this order.
INVESTIGATIONS = [
    ("0. Data quality", data_quality.run, True),
    ("1. How does speed zone influence crash severity?", inv1_speed_zone.run, False),
    ("2. How does crash frequency vary by hour of day and day of week?",
     inv2_time_of_day.run, False),
    ("3. What is the yearly crash trend (2012–2024)?", inv3_yearly_trend.run, False),
    ("4. How does light condition relate to the fatality rate?",
     inv4_light_condition.run, False),
    ("5. Where are the fatal and serious crash hotspots?", inv5_hotspots.run, False),
    ("6. Statistical tests: what predicts a fatal or serious crash?",
     inv6_severity_model.run, False),
    ("7. Deep investigation: does the speed-zone effect hold after adjustment?",
     deep_investigation.run, False),
]


def main():
    set_style()

    start = time.time()
    print("Loading data...")
    raw = load_raw()
    df = clean(raw)
    n_incomplete = int(df["INCOMPLETE_PERIOD"].sum())
    print(f"Loaded {len(df):,} crashes x {df.shape[1]} columns "
          f"({n_incomplete:,} in the incomplete period, flagged INCOMPLETE_PERIOD)")

    for name, run, needs_raw in INVESTIGATIONS:
        print(f"\n=== {name} ===")
        paths = run(df, raw) if needs_raw else run(df)
        for path in paths:
            print(f"  saved {path.relative_to(path.parent.parent)}")

    print(f"\nDone in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
