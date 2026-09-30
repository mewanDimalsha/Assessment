"""Investigation 1: How does speed zone influence crash severity?

Measures, for each posted speed limit, how many crashes there were and what
share were fatal or serious injury (FSI) crashes.

Outputs:
  figures/inv1_speed_zone_severity.png - stacked bar chart of % fatal / % serious
  results/inv1.csv                     - summary table per speed zone
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

from src.utils import SEVERITY_COLOURS, label_axes, save_fig, save_table

FATAL = "Fatal accident"
SERIOUS = "Serious injury accident"
OTHER = "Other injury accident"
NON_INJURY = "Non injury accident"

# Speed zones with fewer crashes than this give unreliable percentages
# (a single fatal crash moves the rate by several points). They are kept in
# the CSV, flagged, but left out of the chart and the chi-square test.
MIN_CRASHES = 100
# Highest/lowest comparisons in the printed key numbers only use zones with at
# least this many crashes, so the quoted ratios aren't driven by a handful of
# fatal crashes (30 km/h zones have 2 fatal crashes, 90 km/h have 21).
MIN_CRASHES_FOR_QUOTE = 1000


def wilson_ci(successes, n, z=1.96):
    """95% Wilson confidence interval for a proportion, in %.
    Better than the normal approximation when the proportion is small
    (e.g. % fatal is only 1-4%)."""
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (centre - half) * 100, (centre + half) * 100


def preprocess(df):
    """Select the crashes used for this investigation, printing each step."""
    n_start = len(df)
    print(f"  Starting rows: {n_start:,}")

    # Step 1 - Drop the incomplete period (Sep 2025 onwards).
    # Those months are only partly entered and severe crashes are entered
    # first, so they are skewed towards FSI crashes (53% FSI vs 37% for the
    # rest of the data). Keeping them would inflate severity.
    incomplete = df["INCOMPLETE_PERIOD"]
    fsi = df["SEVERITY"].isin([FATAL, SERIOUS])
    print(f"  Step 1: removed {int(incomplete.sum()):,} crashes from the incomplete "
          f"period (FSI share {fsi[incomplete].mean():.1%} vs "
          f"{fsi[~incomplete].mean():.1%} in complete months)")
    df = df[~incomplete]

    # Step 2 - Remove unknown / non-numeric speed zones.
    # "Not known" (already set to NaN by clean()), "Other speed limit" and
    # "Camping grounds or off road" have no posted limit in km/h, so they
    # can't be placed on the speed scale. SPEED_ZONE_KMH is NaN for all three.
    unknown = df["SPEED_ZONE_KMH"].isna()
    breakdown = df.loc[unknown, "SPEED_ZONE"].fillna("Not known").value_counts()
    print(f"  Step 2: removed {int(unknown.sum()):,} crashes with unknown speed "
          f"zone ({unknown.mean():.1%} of remaining rows):")
    for label, n in breakdown.items():
        print(f"            {label:<30} {n:>7,}")
    df = df[~unknown]

    # Step 3 - Remove the 4 'Non injury accident' records.
    # The dataset only contains injury crashes (non-injury crashes are not
    # recorded), so these few records are out of scope and would add a
    # near-empty severity category.
    non_injury = df["SEVERITY"] == NON_INJURY
    print(f"  Step 3: removed {int(non_injury.sum()):,} non-injury crashes "
          "(dataset only covers injury crashes)")
    df = df[~non_injury]

    print(f"  Rows used: {len(df):,} ({len(df) / n_start:.1%} of all rows)")
    return df


def summarise(df):
    """Crash count, severity counts and % fatal / serious per speed zone."""
    counts = pd.crosstab(df["SPEED_ZONE_KMH"], df["SEVERITY"])
    counts = counts.reindex(columns=[FATAL, SERIOUS, OTHER], fill_value=0)

    table = pd.DataFrame({
        "speed_zone_kmh": counts.index.astype(int),
        "crashes": counts.sum(axis=1).values,
        "fatal": counts[FATAL].values,
        "serious": counts[SERIOUS].values,
        "other_injury": counts[OTHER].values,
    })
    table["pct_fatal"] = table["fatal"] / table["crashes"] * 100
    table["pct_serious"] = table["serious"] / table["crashes"] * 100
    table["pct_fsi"] = table["pct_fatal"] + table["pct_serious"]
    table["fsi_ci_low"], table["fsi_ci_high"] = wilson_ci(
        table["fatal"] + table["serious"], table["crashes"])
    table["fatal_ci_low"], table["fatal_ci_high"] = wilson_ci(
        table["fatal"], table["crashes"])
    table["pct_of_all_crashes"] = table["crashes"] / table["crashes"].sum() * 100
    table["pct_of_all_fatal"] = table["fatal"] / table["fatal"].sum() * 100
    table["low_n"] = table["crashes"] < MIN_CRASHES
    return table.round(2)


def plot(table):
    """Stacked bar: % fatal (bottom) + % serious (top) = % FSI per speed zone."""
    t = table[~table["low_n"]]
    x = np.arange(len(t))
    labels = [f"{s}\n(n={n:,})" for s, n in zip(t["speed_zone_kmh"], t["crashes"])]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, t["pct_fatal"], color=SEVERITY_COLOURS[FATAL], label="Fatal")
    ax.bar(x, t["pct_serious"], bottom=t["pct_fatal"],
           color=SEVERITY_COLOURS[SERIOUS], label="Serious injury")
    # 95% CI for the combined FSI share.
    ax.errorbar(x, t["pct_fsi"],
                yerr=[t["pct_fsi"] - t["fsi_ci_low"], t["fsi_ci_high"] - t["pct_fsi"]],
                fmt="none", ecolor="black", capsize=3, linewidth=1,
                label="95% CI (fatal + serious)")
    for xi, fsi, ci_high, fatal in zip(x, t["pct_fsi"], t["fsi_ci_high"], t["pct_fatal"]):
        ax.text(xi, ci_high + 0.8, f"{fsi:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        # Fatal segments are thin, so label them just above the segment.
        ax.text(xi, fatal + 0.6, f"fatal {fatal:.1f}%", ha="center", va="bottom",
                fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, t["fsi_ci_high"].max() + 6)
    label_axes(ax, "Share of crashes that were fatal or serious injury, by speed zone",
               "Posted speed limit (km/h)", "Crashes (% of crashes in speed zone)")
    ax.legend(loc="upper left")

    excluded = table[table["low_n"]]
    note = "Victoria, Jan 2012 – Aug 2025. Bold = fatal + serious total."
    if len(excluded):
        note += " Excluded (<{} crashes): {}.".format(
            MIN_CRASHES, ", ".join(f"{s} km/h (n={n})" for s, n in
                                   zip(excluded["speed_zone_kmh"], excluded["crashes"])))
    fig.text(0.01, -0.02, note, fontsize=9, ha="left")
    return save_fig(fig, "inv1_speed_zone_severity")


def association_test(df, table):
    """Chi-square test of independence between speed zone and severity,
    with Cramér's V as the effect size."""
    zones = table.loc[~table["low_n"], "speed_zone_kmh"]
    sub = df[df["SPEED_ZONE_KMH"].isin(zones)]
    observed = pd.crosstab(sub["SPEED_ZONE_KMH"], sub["SEVERITY"])
    chi2, p, dof, _ = chi2_contingency(observed)
    cramers_v = np.sqrt(chi2 / (observed.values.sum() * (min(observed.shape) - 1)))
    return chi2, p, dof, cramers_v


def print_key_numbers(df, table, chi2, p, dof, cramers_v):
    t = table[~table["low_n"]].set_index("speed_zone_kmh")
    overall_fsi = df["SEVERITY"].isin([FATAL, SERIOUS]).mean() * 100
    overall_fatal = (df["SEVERITY"] == FATAL).mean() * 100

    print("\n  Key numbers:")
    print(f"  - {len(df):,} injury crashes with a known speed zone, "
          f"Jan 2012 – Aug 2025")
    print(f"  - Overall: {overall_fsi:.1f}% fatal or serious, {overall_fatal:.2f}% fatal")

    print("\n  Per speed zone (% FSI [95% CI], % fatal [95% CI]):")
    for s, r in t.iterrows():
        print(f"    {s:>3} km/h  n={int(r.crashes):>6,}  "
              f"FSI {r.pct_fsi:5.1f}% [{r.fsi_ci_low:.1f}–{r.fsi_ci_high:.1f}]  "
              f"fatal {r.pct_fatal:4.2f}% [{r.fatal_ci_low:.2f}–{r.fatal_ci_high:.2f}]")

    big = t[t["crashes"] >= MIN_CRASHES_FOR_QUOTE]
    print(f"\n  Comparisons below use zones with >= {MIN_CRASHES_FOR_QUOTE:,} crashes "
          f"({', '.join(str(s) for s in big.index)} km/h):")
    hi, lo = big["pct_fatal"].idxmax(), big["pct_fatal"].idxmin()
    print(f"  - Highest fatal share: {hi} km/h ({t.loc[hi, 'pct_fatal']:.2f}%); "
          f"lowest: {lo} km/h ({t.loc[lo, 'pct_fatal']:.2f}%) - "
          f"{t.loc[hi, 'pct_fatal'] / t.loc[lo, 'pct_fatal']:.1f}x higher")
    hi, lo = big["pct_fsi"].idxmax(), big["pct_fsi"].idxmin()
    print(f"  - Highest FSI share: {hi} km/h ({t.loc[hi, 'pct_fsi']:.1f}%); "
          f"lowest: {lo} km/h ({t.loc[lo, 'pct_fsi']:.1f}%)")

    fast = table["speed_zone_kmh"] >= 80
    print(f"  - Zones of 80 km/h and above: {table.loc[fast, 'pct_of_all_crashes'].sum():.1f}% "
          f"of crashes but {table.loc[fast, 'pct_of_all_fatal'].sum():.1f}% of fatal crashes")
    busiest = t["crashes"].idxmax()
    print(f"  - Most crashes: {busiest} km/h zones "
          f"({t.loc[busiest, 'pct_of_all_crashes']:.1f}% of all crashes)")
    print(f"  - Chi-square test (speed zone x severity): chi2({dof}) = {chi2:,.1f}, "
          f"p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}, Cramér's V = {cramers_v:.3f}")
    low = table[table["low_n"]]
    for _, r in low.iterrows():
        print(f"  - Excluded from chart and test: {r.speed_zone_kmh} km/h "
              f"(only {r.crashes} crashes)")
    print("    Note: with this many crashes almost any difference is 'significant';"
          " Cramér's V shows how strong the association is.")


def run(df):
    """Run investigation 1. Returns the list of saved file paths."""
    data = preprocess(df)
    table = summarise(data)
    fig_path = plot(table)
    csv_path = save_table(table, "inv1")
    chi2, p, dof, cramers_v = association_test(data, table)
    print_key_numbers(data, table, chi2, p, dof, cramers_v)
    return [fig_path, csv_path]
