"""Investigation 0: data quality figures.

Describes the data before any analysis: coverage over time, missing values,
'unknown' codes, location coverage and the distribution of key categories.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import (INCOMPLETE_FROM, ROUTE_SENTINELS,
                             UNKNOWN_BY_COLUMN, UNKNOWN_LABELS)
from src.utils import COLOURS, SEVERITY_COLOURS, label_axes, save_fig

SEVERITY_ORDER = ["Other injury accident", "Serious injury accident",
                  "Fatal accident", "Non injury accident"]


def crashes_per_month(df):
    monthly = df["ACCIDENT_DATE"].dt.to_period("M").value_counts().sort_index()
    x = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, monthly.values, color=COLOURS[0], linewidth=1.5,
            label="Crashes recorded")
    end = x.max() + pd.offsets.MonthBegin(1)
    ax.axvspan(INCOMPLETE_FROM, end, color="grey", alpha=0.25,
               label=f"Incomplete data (from {INCOMPLETE_FROM:%b %Y})")
    ax.set_ylim(bottom=0)
    label_axes(ax, "Crashes recorded per month, Victoria 2012–2026",
               "Month (year)", "Crashes (count per month)")
    ax.legend(loc="lower left")
    return save_fig(fig, "dq_01_crashes_per_month")


def missing_values(raw):
    counts = raw.isna().sum()
    counts = counts[counts > 0].sort_values()
    pct = counts / len(raw) * 100

    fig, ax = plt.subplots(figsize=(10, 0.4 * len(pct) + 1.5))
    bars = ax.barh(pct.index, pct.values, color=COLOURS[0])
    for bar, n, p in zip(bars, counts.values, pct.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{n:,} ({p:.2g}%)", va="center", fontsize=9)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    label_axes(ax, "Missing (blank) values per column",
               "Missing values (% of crashes)", "Column")
    return save_fig(fig, "dq_02_missing_values")


def unknown_labels(raw):
    rows = []
    for col in raw.columns:
        labels = UNKNOWN_LABELS + UNKNOWN_BY_COLUMN.get(col, [])
        n = int(raw[col].isin(labels).sum())
        if n:
            rows.append((col, n))
    if "ROAD_ROUTE_1" in raw.columns:
        n = int(raw["ROAD_ROUTE_1"].isin(ROUTE_SENTINELS).sum())
        rows.append(("ROAD_ROUTE_1 (9999 / -1)", n))
    counts = pd.Series(dict(rows)).sort_values()
    pct = counts / len(raw) * 100

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(counts) + 1.5))
    bars = ax.barh(counts.index, pct.values, color=COLOURS[1])
    for bar, n, p in zip(bars, counts.values, pct.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{n:,} ({p:.1f}%)", va="center", fontsize=9)
    ax.set_xlim(0, pct.max() * 1.3)
    ax.grid(axis="y", visible=False)
    label_axes(ax, "'Unknown' codes per column (treated as missing)",
               "Crashes with unknown value (% of crashes)", "Column")
    return save_fig(fig, "dq_03_unknown_labels")


def crash_locations(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    # Draw the most common severity first so fatal crashes stay visible on top.
    for severity in SEVERITY_ORDER:
        sub = df[df["SEVERITY"] == severity]
        if sub.empty:
            continue
        size = 2 if severity == "Fatal accident" else 0.5
        ax.scatter(sub["LONGITUDE"], sub["LATITUDE"], s=size, alpha=0.6,
                   color=SEVERITY_COLOURS[severity], linewidths=0,
                   label=f"{severity} (n={len(sub):,})", rasterized=True)
    # Correct the aspect ratio for latitude so the map is not stretched.
    ax.set_aspect(1 / np.cos(np.radians(df["LATITUDE"].mean())))
    label_axes(ax, "Crash locations by severity, Victoria 2012–2026",
               "Longitude (degrees east)", "Latitude (degrees, negative = south)")
    # Empty area in the north-east keeps the legend off the data.
    legend = ax.legend(loc="upper right", title="Severity")
    for handle in legend.legend_handles:
        handle.set_sizes([30])
        handle.set_alpha(1)
    missing = int(df["LATITUDE"].isna().sum())
    ax.text(0.99, 0.01, f"{missing} crashes without coordinates not shown",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    return save_fig(fig, "dq_04_crash_locations")


def categories(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sev = df["SEVERITY"].value_counts().reindex(SEVERITY_ORDER).dropna()
    ax = axes[0, 0]
    ax.bar(sev.index.str.replace(" accident", ""), sev.values,
           color=[SEVERITY_COLOURS[s] for s in sev.index])
    label_axes(ax, "Severity", "Crash severity", "Crashes (count)")

    speed = df["SPEED_ZONE_KMH"].value_counts().sort_index()
    ax = axes[0, 1]
    ax.bar(speed.index.astype(int).astype(str), speed.values, color=COLOURS[0])
    label_axes(ax, f"Speed limit ({int(df['SPEED_ZONE_KMH'].isna().sum()):,} "
               "unknown/other excluded)", "Speed limit (km/h)", "Crashes (count)")

    types = df["ACCIDENT_TYPE"].str.capitalize().value_counts().sort_values()
    ax = axes[1, 0]
    ax.barh(types.index, types.values, color=COLOURS[2])
    ax.grid(axis="y", visible=False)
    label_axes(ax, "Crash type", "Crashes (count)", "Crash type")

    days = df["DAY_OF_WEEK"].value_counts().sort_index()
    ax = axes[1, 1]
    ax.bar(days.index.astype(str).str[:3], days.values, color=COLOURS[4])
    label_axes(ax, "Day of week", "Day of week", "Crashes (count)")

    # Thousands separators on the count axis (x for the horizontal chart).
    thousands = plt.FuncFormatter(lambda v, _: f"{v:,.0f}")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.yaxis.set_major_formatter(thousands)
    axes[1, 0].xaxis.set_major_formatter(thousands)
    fig.suptitle("Distribution of key crash attributes, Victoria 2012–2026",
                 fontsize=15, fontweight="bold")
    return save_fig(fig, "dq_05_categories")


def run(df, raw):
    """Create all data quality figures. Returns the list of saved paths."""
    return [
        crashes_per_month(df),
        missing_values(raw),
        unknown_labels(raw),
        crash_locations(df),
        categories(df),
    ]
