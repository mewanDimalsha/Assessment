"""Investigation 3: What is the yearly crash trend (excluding incomplete years)?

Tracks crashes, their severity mix and deaths per year for the full years
2012-2024. 2025 is only complete up to August, so it is compared with earlier
years on a like-for-like January-August basis instead.

Outputs:
  figures/inv3_yearly_trend.png - stacked bars per year + Jan-Aug comparison
  results/inv3.csv              - summary table, one row per year 2012-2025
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import linregress

from src.data_loader import INCOMPLETE_FROM
from src.inv1_speed_zone import FATAL, NON_INJURY, OTHER, SERIOUS, wilson_ci
from src.utils import COLOURS, SEVERITY_COLOURS, label_axes, save_fig, save_table

# Last calendar year with all 12 months fully entered (2024).
LAST_FULL_YEAR = INCOMPLETE_FROM.year - 1
# Year that is complete only up to the incomplete cut-off (2025, Jan-Aug).
PARTIAL_YEAR = INCOMPLETE_FROM.year
# Months that are complete in every year, used for the like-for-like panel.
JAN_AUG = list(range(1, INCOMPLETE_FROM.month))
# Melbourne's COVID-19 lockdowns sharply reduced traffic in these years.
COVID_YEARS = [2020, 2021]


def preprocess(df):
    """Return (all_years, full_years) crash sets, printing each step."""
    n_start = len(df)
    print(f"  Starting rows: {n_start:,}")

    # Step 1 - Remove the 'Non injury accident' records.
    # The dataset only contains injury crashes, so these few records are out
    # of scope (same as investigations 1 and 2). Done first so the Jan-Aug
    # comparison and the main trend use the same set of crashes.
    non_injury = df["SEVERITY"] == NON_INJURY
    print(f"  Step 1: removed {int(non_injury.sum()):,} non-injury crashes "
          "(dataset only covers injury crashes)")
    df = df[~non_injury]

    # Year is never unknown: every ACCIDENT_DATE parses (checked in explore.py),
    # so there is no 'unknown' removal step here.
    print(f"  (No unknown years: {int(df['YEAR'].isna().sum())} unparseable dates)")

    # Step 2 - Remove incomplete years from the main trend.
    # 2026 only has January, and 2025 is under-counted from September because
    # recent crashes are still being entered. Their full-year totals would show
    # a false drop. 2025 is kept for the Jan-Aug panel, where only months that
    # are complete in every year are compared.
    partial = df["YEAR"] > LAST_FULL_YEAR
    p = df[df["YEAR"] == PARTIAL_YEAR]
    p_jan_aug = int(p["MONTH"].isin(JAN_AUG).sum())
    print(f"  Step 2: removed {int(partial.sum()):,} crashes from {PARTIAL_YEAR} onwards "
          f"from the main trend ({PARTIAL_YEAR}: {len(p):,} crashes in the full year, "
          f"of which {p_jan_aug:,} in Jan–Aug - kept for the Jan–Aug comparison)")
    full = df[~partial]
    jan_aug = df[df["MONTH"].isin(JAN_AUG) & (df["YEAR"] <= PARTIAL_YEAR)]

    print(f"  Rows used: {len(full):,} for the {full['YEAR'].min()}–{LAST_FULL_YEAR} "
          f"trend; {len(jan_aug):,} for the Jan–Aug comparison "
          f"({jan_aug['YEAR'].min()}–{PARTIAL_YEAR})")
    return full, jan_aug


def summarise(full, jan_aug):
    """One row per year. Full-year columns are blank for the partial year."""
    by_year = full.groupby("YEAR")
    counts = pd.crosstab(full["YEAR"], full["SEVERITY"]).reindex(
        columns=[FATAL, SERIOUS, OTHER], fill_value=0)

    table = pd.DataFrame({
        "crashes": by_year.size(),
        "fatal": counts[FATAL],
        "serious": counts[SERIOUS],
        "other_injury": counts[OTHER],
        "deaths": by_year["FATALITY"].sum(),
        "serious_injuries": by_year["SERIOUSINJURY"].sum(),
    })
    table["pct_fatal"] = table["fatal"] / table["crashes"] * 100
    table["pct_fsi"] = (table["fatal"] + table["serious"]) / table["crashes"] * 100
    table["fsi_ci_low"], table["fsi_ci_high"] = wilson_ci(
        table["fatal"] + table["serious"], table["crashes"])
    table["pct_change_vs_prev_year"] = table["crashes"].pct_change() * 100
    table["index_2012"] = table["crashes"] / table["crashes"].iloc[0] * 100

    # Add the Jan-Aug count for every year, including the partial year.
    table = table.reindex(sorted(jan_aug["YEAR"].unique()))
    table["jan_aug_crashes"] = jan_aug.groupby("YEAR").size()
    table["full_year"] = table.index <= LAST_FULL_YEAR

    table = table.rename_axis("year").reset_index()
    # Keep counts as whole numbers even though the partial year is blank.
    for col in ["crashes", "fatal", "serious", "other_injury", "deaths",
                "serious_injuries", "jan_aug_crashes"]:
        table[col] = table[col].astype("Int64")
    cols = ["year", "full_year", "crashes", "fatal", "serious", "other_injury",
            "pct_fatal", "pct_fsi", "fsi_ci_low", "fsi_ci_high", "deaths",
            "serious_injuries", "pct_change_vs_prev_year", "index_2012",
            "jan_aug_crashes"]
    return table[cols].round(2)


def plot(table):
    full = table[table["full_year"]]
    years = full["year"].astype(int)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 11),
                                      gridspec_kw={"height_ratios": [3, 2]})

    # --- Top: stacked bars of crashes by severity, FSI share on second axis.
    bottom_of_bar = np.zeros(len(full))
    for col, sev, label in [("fatal", FATAL, "Fatal"),
                            ("serious", SERIOUS, "Serious injury"),
                            ("other_injury", OTHER, "Other injury")]:
        top.bar(years, full[col], bottom=bottom_of_bar,
                color=SEVERITY_COLOURS[sev], label=label, width=0.75)
        bottom_of_bar += full[col].values
    for y in COVID_YEARS:
        top.axvspan(y - 0.5, y + 0.5, color="grey", alpha=0.15, zorder=0)

    share = top.twinx()
    share.plot(years, full["pct_fsi"], color="black", marker="o", markersize=4,
               linewidth=1.5, label="Fatal or serious (% of crashes)")
    share.fill_between(years, full["fsi_ci_low"], full["fsi_ci_high"],
                       color="black", alpha=0.12, linewidth=0,
                       label="95% CI (fatal or serious %)")
    share.set_ylim(0, 60)
    share.set_ylabel("Fatal or serious (% of crashes)")
    share.grid(False)
    share.spines["right"].set_visible(True)

    top.set_ylim(0, bottom_of_bar.max() * 1.25)
    top.set_xticks(years)
    label_axes(top, f"Injury crashes per year by severity, Victoria "
               f"{years.min()}–{LAST_FULL_YEAR}",
               "Year", "Crashes (count per year)")
    # One legend for both axes, plus the COVID shading.
    handles, labels = top.get_legend_handles_labels()
    h2, l2 = share.get_legend_handles_labels()
    handles += h2 + [Patch(color="grey", alpha=0.15)]
    labels += l2 + ["COVID-19 lockdown years"]
    top.legend(handles, labels, loc="upper left", ncol=3, fontsize=9)

    # --- Bottom: Jan-Aug crashes for every year, partial year highlighted.
    colours = [COLOURS[1] if y == PARTIAL_YEAR else COLOURS[0] for y in table["year"]]
    bottom.bar(table["year"], table["jan_aug_crashes"], color=colours, width=0.75)
    avg = full["jan_aug_crashes"].mean()
    bottom.axhline(avg, color="black", linestyle="--", linewidth=1)
    bottom.set_xticks(table["year"])
    bottom.set_ylim(0, table["jan_aug_crashes"].max() * 1.2)
    label_axes(bottom, "Like-for-like comparison: crashes from January to August",
               "Year", "Crashes, Jan–Aug (count)")
    bottom.legend(handles=[
        Patch(color=COLOURS[0], label=f"{years.min()}–{LAST_FULL_YEAR}"),
        Patch(color=COLOURS[1], label=f"{PARTIAL_YEAR} (latest data)"),
        plt.Line2D([], [], color="black", linestyle="--",
                   label=f"{years.min()}–{LAST_FULL_YEAR} average ({avg:,.0f})"),
    ], loc="upper left", ncol=3, fontsize=9)

    thousands = plt.FuncFormatter(lambda v, _: f"{v:,.0f}")
    top.yaxis.set_major_formatter(thousands)
    bottom.yaxis.set_major_formatter(thousands)
    fig.text(0.01, -0.01, f"Injury crashes. {PARTIAL_YEAR} excluded from the top "
             f"panel: data under-counted from {INCOMPLETE_FROM:%b %Y}.",
             fontsize=9, ha="left")
    return save_fig(fig, "inv3_yearly_trend")


def trend(x, y):
    fit = linregress(x, y)
    return fit.slope, fit.rvalue ** 2, fit.pvalue


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def print_key_numbers(table):
    full = table[table["full_year"]].set_index("year")
    first, last = full.index.min(), full.index.max()

    print("\n  Key numbers:")
    c = full["crashes"]
    print(f"  - Crashes: {c[first]:,.0f} in {first} vs {c[last]:,.0f} in {last} "
          f"({(c[last] / c[first] - 1) * 100:+.1f}%)")
    print(f"  - Peak year: {c.idxmax()} ({c.max():,.0f}); lowest year: "
          f"{c.idxmin()} ({c.min():,.0f})")
    print(f"  - COVID dip: 2020 had {(c[2020] / c[2019] - 1) * 100:+.1f}% crashes vs 2019; "
          f"2022 was {(c[2022] / c[2019] - 1) * 100:+.1f}% vs 2019")

    slope, r2, p = trend(full.index, c)
    print(f"  - Linear trend {first}–{last}: {slope:+.0f} crashes per year "
          f"(R² = {r2:.2f}, p {fmt_p(p)})")
    no_covid = full.drop(COVID_YEARS)
    slope, r2, p = trend(no_covid.index, no_covid["crashes"])
    print(f"  - Linear trend excluding {COVID_YEARS[0]}–{COVID_YEARS[-1]}: "
          f"{slope:+.0f} crashes per year (R² = {r2:.2f}, p {fmt_p(p)})")
    print("    Note: only 13 yearly points - describe the trend, don't forecast from it.")

    f = full
    print(f"  - Fatal or serious share: {f.loc[first, 'pct_fsi']:.1f}% "
          f"[{f.loc[first, 'fsi_ci_low']:.1f}–{f.loc[first, 'fsi_ci_high']:.1f}] in {first} vs "
          f"{f.loc[last, 'pct_fsi']:.1f}% [{f.loc[last, 'fsi_ci_low']:.1f}–"
          f"{f.loc[last, 'fsi_ci_high']:.1f}] in {last}; peak "
          f"{f['pct_fsi'].max():.1f}% in {f['pct_fsi'].idxmax()}")
    slope, r2, p = trend(f.index, f["pct_fsi"])
    print(f"    trend {slope:+.2f} percentage points per year (R² = {r2:.2f}, p {fmt_p(p)})")

    d = full["deaths"]
    print(f"  - Deaths: {d[first]:,.0f} in {first} vs {d[last]:,.0f} in {last}; "
          f"lowest {d.min():,.0f} ({d.idxmin()}), highest {d.max():,.0f} ({d.idxmax()}); "
          f"total {d.sum():,.0f} over {len(d)} years")

    ja = table.set_index("year")["jan_aug_crashes"]
    recent = ja.loc[last - 2:last].mean()
    print(f"  - Jan–Aug {PARTIAL_YEAR}: {ja[PARTIAL_YEAR]:,.0f} crashes vs "
          f"{recent:,.0f} average for Jan–Aug {last - 2}–{last} "
          f"({(ja[PARTIAL_YEAR] / recent - 1) * 100:+.1f}%); highest Jan–Aug "
          f"total: {ja.idxmax()} ({ja.max():,.0f})")


def run(df):
    """Run investigation 3. Returns the list of saved file paths."""
    full, jan_aug = preprocess(df)
    table = summarise(full, jan_aug)
    paths = [plot(table), save_table(table, "inv3")]
    print_key_numbers(table)
    return paths
