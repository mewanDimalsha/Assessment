"""Investigation 2: How does crash frequency vary by hour of day and day of week?

Counts crashes in each of the 168 hour-of-week slots (7 days x 24 hours),
compares weekdays with weekends, and checks whether the fatal / serious injury
(FSI) share changes with time of day.

Outputs:
  figures/inv2_heatmap_hour_day.png - hour x day heatmap of crash frequency
  figures/inv2_hourly_profile.png   - weekday vs weekend crashes and FSI share by hour
  results/inv2.csv                  - summary table, one row per day x hour
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare

from src.data_loader import DAY_ORDER, INCOMPLETE_FROM
from src.inv1_speed_zone import FATAL, NON_INJURY, SERIOUS, wilson_ci
from src.utils import COLOURS, label_axes, save_fig, save_table

WEEKEND = ["Saturday", "Sunday"]
# Night = 22:00-05:59, day = 06:00-21:59.
NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4, 5]
# Weekday afternoon peak used for a quotable share (15:00-18:59).
PEAK_HOURS = [15, 16, 17, 18]


def preprocess(df):
    """Select the crashes used for this investigation, printing each step."""
    n_start = len(df)
    print(f"  Starting rows: {n_start:,}")

    # Step 1 - Drop the incomplete period (Sep 2025 onwards).
    # Those months are only partly entered and severe crashes are entered
    # first, so they are skewed towards FSI crashes. They would also add a few
    # extra weekdays with too few crashes, lowering the averages.
    incomplete = df["INCOMPLETE_PERIOD"]
    fsi = df["SEVERITY"].isin([FATAL, SERIOUS])
    print(f"  Step 1: removed {int(incomplete.sum()):,} crashes from the incomplete "
          f"period (FSI share {fsi[incomplete].mean():.1%} vs "
          f"{fsi[~incomplete].mean():.1%} in complete months)")
    df = df[~incomplete]

    # Step 2 - Remove crashes with an unknown hour.
    # clean() already reads the 85 times stored as "HH:MM:SS.0000000", so
    # none should be missing. The 113 crashes at exactly 00:00 are kept:
    # that is fewer than the neighbouring half-hours (23:30 has 491,
    # 00:30 has 368), so midnight is not being used as an "unknown" code.
    unknown = df["HOUR"].isna() | df["DAY_OF_WEEK"].isna()
    print(f"  Step 2: removed {int(unknown.sum()):,} crashes with unknown hour "
          "or day (midnight 00:00 kept - not a placeholder)")
    df = df[~unknown]

    # Step 3 - Remove the 'Non injury accident' records.
    # The dataset only contains injury crashes, so these few records are out
    # of scope (same as investigation 1).
    non_injury = df["SEVERITY"] == NON_INJURY
    print(f"  Step 3: removed {int(non_injury.sum()):,} non-injury crashes "
          "(dataset only covers injury crashes)")
    df = df[~non_injury].copy()
    df["HOUR"] = df["HOUR"].astype(int)

    print(f"  Rows used: {len(df):,} ({len(df) / n_start:.1%} of all rows)")
    return df


def days_per_weekday(df):
    """Number of Mondays, Tuesdays, ... in the study period. Used to turn
    counts into averages, so each day is compared fairly."""
    start = df["ACCIDENT_DATE"].min()
    end = INCOMPLETE_FROM - pd.Timedelta(days=1)
    dates = pd.date_range(start, end, freq="D")
    return pd.Series(dates.day_name()).value_counts().reindex(DAY_ORDER)


def summarise(df):
    """One row per (day, hour): counts, averages and severity shares."""
    grouped = df.groupby(["DAY_OF_WEEK", "HOUR"], observed=False)
    table = pd.DataFrame({
        "crashes": grouped.size(),
        "fatal": grouped["SEVERITY"].apply(lambda s: (s == FATAL).sum()),
        "serious": grouped["SEVERITY"].apply(lambda s: (s == SERIOUS).sum()),
    }).reset_index()
    table.columns = ["day_of_week", "hour", "crashes", "fatal", "serious"]
    table["day_of_week"] = table["day_of_week"].astype(str)

    n_days = days_per_weekday(df)
    # Each (day, hour) slot happens once a week, so this is crashes per week.
    table["crashes_per_week"] = table["crashes"] / table["day_of_week"].map(n_days)
    table["pct_fatal"] = table["fatal"] / table["crashes"] * 100
    table["pct_fsi"] = (table["fatal"] + table["serious"]) / table["crashes"] * 100
    table["fsi_ci_low"], table["fsi_ci_high"] = wilson_ci(
        table["fatal"] + table["serious"], table["crashes"])
    table["pct_of_all_crashes"] = table["crashes"] / table["crashes"].sum() * 100
    return table.round(3)


def plot_heatmap(table):
    grid = table.pivot(index="day_of_week", columns="hour",
                       values="crashes_per_week").reindex(DAY_ORDER)

    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(grid.values, cmap="YlOrRd", aspect="auto")
    # Annotate each cell; dark text on light cells, white on dark cells.
    threshold = grid.values.max() * 0.6
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if v > threshold else "black")

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_yticks(range(7))
    ax.set_yticklabels(DAY_ORDER)
    ax.grid(False)
    label_axes(ax, "Crash frequency by hour of day and day of week, Victoria",
               "Hour of day (24 h clock, start of hour)", "Day of week")
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Crashes (average per week in this hour)")
    fig.text(0.01, -0.03, "Injury crashes, Jan 2012 – Aug 2025. Each cell = "
             "average number of crashes in that one-hour slot per week.",
             fontsize=9, ha="left")
    return save_fig(fig, "inv2_heatmap_hour_day")


def hourly_by_period(df, n_days):
    """Weekday vs weekend: average crashes per day and FSI share, by hour."""
    rows = {}
    for label, days in [("Weekday (Mon–Fri)", DAY_ORDER[:5]), ("Weekend (Sat–Sun)", WEEKEND)]:
        sub = df[df["DAY_OF_WEEK"].isin(days)]
        crashes = sub.groupby("HOUR").size().reindex(range(24), fill_value=0)
        fsi = sub["SEVERITY"].isin([FATAL, SERIOUS]).groupby(sub["HOUR"]).sum() \
            .reindex(range(24), fill_value=0)
        low, high = wilson_ci(fsi, crashes)
        rows[label] = pd.DataFrame({
            "per_day": crashes / n_days[days].sum(),
            "pct_fsi": fsi / crashes * 100,
            "ci_low": low, "ci_high": high,
        })
    return rows


def plot_profile(profiles):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2]})
    colours = [COLOURS[0], COLOURS[3]]
    for (label, p), colour in zip(profiles.items(), colours):
        top.plot(p.index, p["per_day"], marker="o", markersize=4,
                 color=colour, label=label)
        bottom.plot(p.index, p["pct_fsi"], marker="o", markersize=4,
                    color=colour, label=label)
        bottom.fill_between(p.index, p["ci_low"], p["ci_high"],
                            color=colour, alpha=0.2, linewidth=0,
                            label=f"95% CI ({label.split(' ')[0].lower()})")

    # Shade night hours; label it once so it appears in the top legend.
    for ax in (top, bottom):
        ax.axvspan(-0.5, 5.5, color="grey", alpha=0.08,
                   label="Night (22:00–05:59)" if ax is top else None)
        ax.axvspan(21.5, 23.5, color="grey", alpha=0.08)

    top.set_ylim(bottom=0)
    label_axes(top, "Crashes by hour of day: weekdays vs weekends",
               "", "Crashes (average per day)")
    top.legend(loc="upper right")

    label_axes(bottom, "Share of crashes that were fatal or serious injury, by hour",
               "Hour of day (24 h clock, start of hour)",
               "Fatal or serious (% of crashes)")
    bottom.legend(loc="upper right", ncol=2, fontsize=9)
    bottom.set_xticks(range(24))
    bottom.set_xticklabels([f"{h:02d}" for h in range(24)])
    bottom.set_xlim(-0.5, 23.5)
    fig.text(0.01, -0.01, "Injury crashes, Jan 2012 – Aug 2025.", fontsize=9, ha="left")
    return save_fig(fig, "inv2_hourly_profile")


def print_key_numbers(df, table, profiles, n_days):
    fmt_slot = lambda r: f"{r.day_of_week[:3]} {r.hour:02d}:00–{(r.hour + 1) % 24:02d}:00"
    peak = table.loc[table["crashes_per_week"].idxmax()]
    quiet = table.loc[table["crashes_per_week"].idxmin()]

    print("\n  Key numbers:")
    print(f"  - {len(df):,} injury crashes, Jan 2012 – Aug 2025 "
          f"({int(n_days.sum()):,} days)")
    print(f"  - Busiest hour-slot: {fmt_slot(peak)} - {peak.crashes_per_week:.2f} "
          f"crashes per week ({int(peak.crashes):,} in total)")
    print(f"  - Quietest hour-slot: {fmt_slot(quiet)} - {quiet.crashes_per_week:.2f} "
          f"crashes per week ({peak.crashes_per_week / quiet.crashes_per_week:.0f}x fewer)")

    wd, we = profiles["Weekday (Mon–Fri)"], profiles["Weekend (Sat–Sun)"]
    print(f"  - Weekday peak hour: {wd['per_day'].idxmax():02d}:00 "
          f"({wd['per_day'].max():.2f} crashes per day); weekend peak hour: "
          f"{we['per_day'].idxmax():02d}:00 ({we['per_day'].max():.2f} per day)")
    print(f"  - Average crashes per day: weekday "
          f"{wd['per_day'].sum():.1f}, weekend {we['per_day'].sum():.1f}")

    weekday = table["day_of_week"].isin(DAY_ORDER[:5])
    peak_mask = weekday & table["hour"].isin(PEAK_HOURS)
    share_of_hours = len(PEAK_HOURS) * 5 / 168 * 100
    print(f"  - Weekday afternoon peak (15:00–18:59, Mon–Fri): "
          f"{table.loc[peak_mask, 'pct_of_all_crashes'].sum():.1f}% of crashes "
          f"in {share_of_hours:.1f}% of the week's hours")

    per_day = table.groupby("day_of_week")["crashes"].sum() / n_days
    share = table.groupby("day_of_week")["crashes"].sum() / table["crashes"].sum() * 100
    print(f"  - Busiest day: {per_day.idxmax()} ({per_day.max():.1f} crashes/day, "
          f"{share[per_day.idxmax()]:.1f}% of crashes); quietest: {per_day.idxmin()} "
          f"({per_day.min():.1f} crashes/day, {share[per_day.idxmin()]:.1f}%)")

    night = df["HOUR"].isin(NIGHT_HOURS)
    fsi = df["SEVERITY"].isin([FATAL, SERIOUS])
    fatal = df["SEVERITY"] == FATAL
    for label, mask in [("Night (22:00–05:59)", night), ("Day (06:00–21:59)", ~night)]:
        lo, hi = wilson_ci(fsi[mask].sum(), mask.sum())
        print(f"  - {label}: {mask.mean():.1%} of crashes, FSI {fsi[mask].mean():.1%} "
              f"[95% CI {lo:.1f}–{hi:.1f}%], fatal {fatal[mask].mean():.2%}")
    print(f"    -> night crashes are {fatal[night].mean() / fatal[~night].mean():.1f}x "
          "as likely to be fatal as daytime crashes")

    # Are crashes spread evenly over the 168 hour-slots? Expected counts are
    # weighted by how many of each weekday are in the period.
    expected = table["day_of_week"].map(n_days) / n_days.sum() * table["crashes"].sum() / 24
    chi2, p = chisquare(table["crashes"], expected)
    w = np.sqrt(chi2 / table["crashes"].sum())
    print(f"  - Chi-square goodness of fit vs an even spread over 168 hour-slots: "
          f"chi2(167) = {chi2:,.0f}, p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}, "
          f"Cohen's w = {w:.2f} (0.5+ = large effect)")


def run(df):
    """Run investigation 2. Returns the list of saved file paths."""
    data = preprocess(df)
    n_days = days_per_weekday(data)
    table = summarise(data)
    profiles = hourly_by_period(data, n_days)
    paths = [plot_heatmap(table), plot_profile(profiles),
             save_table(table, "inv2")]
    print_key_numbers(data, table, profiles, n_days)
    return paths
