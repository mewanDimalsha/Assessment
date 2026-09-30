"""Investigation 4: How does light condition relate to the fatality rate?

Compares the share of crashes that were fatal (and fatal or serious) across
light conditions, then repeats the comparison within speed bands to check how
much of the difference is explained by unlit roads being high-speed roads.

Outputs:
  figures/inv4_light_condition.png - stacked bar per light condition + fatal
                                     share by light condition within speed bands
  results/inv4.csv                 - summary table per light condition
  results/inv4_by_speed.csv        - fatal share per light condition x speed band
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency

from src.inv1_speed_zone import FATAL, MIN_CRASHES, NON_INJURY, SERIOUS, wilson_ci
from src.utils import SEVERITY_COLOURS, label_axes, save_fig, save_table

# Ordered from most to least light.
LIGHT_ORDER = ["Day", "Dusk/Dawn", "Dark Street lights on",
               "Dark Street lights off", "Dark No street lights"]
SHORT_LABELS = {
    "Day": "Day",
    "Dusk/Dawn": "Dusk/dawn",
    "Dark Street lights on": "Dark,\nlights on",
    "Dark Street lights off": "Dark,\nlights off",
    "Dark No street lights": "Dark,\nno street lights",
}
# Okabe-Ito colours running from light (yellow) to dark (black).
LIGHT_COLOURS = {
    "Day": "#F0E442",
    "Dusk/Dawn": "#E69F00",
    "Dark Street lights on": "#56B4E9",
    "Dark Street lights off": "#0072B2",
    "Dark No street lights": "#000000",
}
# Speed bands: urban streets, outer urban / rural arterials, rural highways.
SPEED_BINS = [0, 60, 90, 200]
SPEED_BANDS = ["≤60 km/h", "70–90 km/h", "≥100 km/h"]

# "Dark Street lights off" has only 426 crashes in total (255 / 108 / 43 per
# speed band), so its within-band rates have very wide intervals. It is shown
# in the left panel and kept in inv4_by_speed.csv, but left out of the
# speed-band panel so it doesn't stretch the scale for the other groups.
SPEED_PANEL_LIGHTS = [l for l in LIGHT_ORDER if l != "Dark Street lights off"]

REFERENCE = "Day"
UNLIT = "Dark No street lights"
LIT = "Dark Street lights on"


def preprocess(df):
    """Select the crashes used for this investigation, printing each step."""
    n_start = len(df)
    print(f"  Starting rows: {n_start:,}")

    # Step 1 - Drop the incomplete period (Sep 2025 onwards).
    # Severe crashes are entered first, so these months over-represent fatal
    # and serious crashes and would inflate the rates (same as inv1 and inv2).
    incomplete = df["INCOMPLETE_PERIOD"]
    print(f"  Step 1: removed {int(incomplete.sum()):,} crashes from the "
          "incomplete period (over-represents severe crashes)")
    df = df[~incomplete]

    # Step 2 - Remove unknown light conditions.
    # clean() has already turned both of these labels into NaN:
    #   "Unk."                       - light condition not recorded at all
    #   "Dark Street lights unknown" - it was dark, but whether street lights
    #                                  were on is not known
    # The second group can't be placed in any of the dark categories being
    # compared (lights on / off / none), so both are removed.
    unknown = df["LIGHT_CONDITION"].isna()
    print(f"  Step 2: removed {int(unknown.sum()):,} crashes with unknown light "
          f"condition ({unknown.mean():.1%} of remaining rows; "
          "'Unk.' and 'Dark Street lights unknown')")
    df = df[~unknown]

    # Step 3 - Remove the 'Non injury accident' records.
    # The dataset only contains injury crashes, so these are out of scope.
    non_injury = df["SEVERITY"] == NON_INJURY
    print(f"  Step 3: removed {int(non_injury.sum()):,} non-injury crashes "
          "(dataset only covers injury crashes)")
    df = df[~non_injury].copy()

    print(f"  Rows used: {len(df):,} ({len(df) / n_start:.1%} of all rows)")
    return df


def rate_ratio(a, n1, c, n2, z=1.96):
    """Ratio of two proportions (a/n1) / (c/n2) with a 95% CI (log method)."""
    ratio = (a / n1) / (c / n2)
    se = np.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n2)
    return ratio, ratio * np.exp(-z * se), ratio * np.exp(z * se)


def summarise(df):
    """One row per light condition."""
    g = df.groupby("LIGHT_CONDITION")
    table = pd.DataFrame({
        "crashes": g.size(),
        "fatal": g["SEVERITY"].apply(lambda s: (s == FATAL).sum()),
        "serious": g["SEVERITY"].apply(lambda s: (s == SERIOUS).sum()),
        "deaths": g["FATALITY"].sum(),
    }).reindex(LIGHT_ORDER)

    table["pct_fatal"] = table["fatal"] / table["crashes"] * 100
    table["pct_serious"] = table["serious"] / table["crashes"] * 100
    table["pct_fsi"] = table["pct_fatal"] + table["pct_serious"]
    table["fatal_ci_low"], table["fatal_ci_high"] = wilson_ci(table["fatal"], table["crashes"])
    table["fsi_ci_low"], table["fsi_ci_high"] = wilson_ci(
        table["fatal"] + table["serious"], table["crashes"])
    table["deaths_per_100_crashes"] = table["deaths"] / table["crashes"] * 100

    ref = table.loc[REFERENCE]
    ratios = [rate_ratio(r.fatal, r.crashes, ref.fatal, ref.crashes)
              for r in table.itertuples()]
    table["fatal_rate_ratio_vs_day"], table["ratio_ci_low"], table["ratio_ci_high"] = zip(*ratios)

    table["pct_of_crashes"] = table["crashes"] / table["crashes"].sum() * 100
    table["pct_of_deaths"] = table["deaths"] / table["deaths"].sum() * 100
    table["low_n"] = table["crashes"] < MIN_CRASHES
    return table.rename_axis("light_condition").reset_index().round(3)


def by_speed_band(df):
    """Fatal share per light condition within each speed band."""
    known = df["SPEED_ZONE_KMH"].notna()
    print(f"  Speed-band breakdown: {int((~known).sum()):,} crashes with unknown "
          "speed zone left out of this part only")
    sub = df[known].copy()
    sub["speed_band"] = pd.cut(sub["SPEED_ZONE_KMH"], SPEED_BINS, labels=SPEED_BANDS)

    g = sub.groupby(["speed_band", "LIGHT_CONDITION"], observed=True)
    table = pd.DataFrame({
        "crashes": g.size(),
        "fatal": g["SEVERITY"].apply(lambda s: (s == FATAL).sum()),
    })
    full_index = pd.MultiIndex.from_product([SPEED_BANDS, LIGHT_ORDER],
                                            names=["speed_band", "light_condition"])
    table = table.reindex(full_index, fill_value=0).reset_index()
    table["pct_fatal"] = table["fatal"] / table["crashes"] * 100
    table["fatal_ci_low"], table["fatal_ci_high"] = wilson_ci(table["fatal"], table["crashes"])
    table["low_n"] = table["crashes"] < MIN_CRASHES
    return table.round(3)


def plot(table, bands):
    fig, (left, right) = plt.subplots(1, 2, figsize=(17, 7),
                                      gridspec_kw={"width_ratios": [1, 1.2]})

    # --- Left: stacked % fatal + % serious per light condition.
    x = np.arange(len(table))
    left.bar(x, table["pct_fatal"], color=SEVERITY_COLOURS[FATAL], label="Fatal")
    left.bar(x, table["pct_serious"], bottom=table["pct_fatal"],
             color=SEVERITY_COLOURS[SERIOUS], label="Serious injury")
    left.errorbar(x, table["pct_fsi"],
                  yerr=[table["pct_fsi"] - table["fsi_ci_low"],
                        table["fsi_ci_high"] - table["pct_fsi"]],
                  fmt="none", ecolor="black", capsize=3, linewidth=1,
                  label="95% CI (fatal + serious)")
    for xi, r in zip(x, table.itertuples()):
        left.text(xi, r.fsi_ci_high + 0.8, f"{r.pct_fsi:.1f}%", ha="center",
                  va="bottom", fontsize=10, fontweight="bold")
        left.text(xi, r.pct_fatal + 0.6, f"fatal {r.pct_fatal:.1f}%", ha="center",
                  va="bottom", fontsize=8)
    left.set_xticks(x)
    left.set_xticklabels([f"{SHORT_LABELS[c]}\n(n={n:,})" for c, n in
                          zip(table["light_condition"], table["crashes"])], fontsize=9)
    left.set_ylim(0, table["fsi_ci_high"].max() + 10)
    label_axes(left, "Fatal and serious injury crashes by light condition",
               "Light condition", "Crashes (% of crashes in light condition)")
    left.legend(loc="upper left", fontsize=9)

    # --- Right: % fatal by light condition within each speed band.
    width = 0.8 / len(SPEED_PANEL_LIGHTS)
    xb = np.arange(len(SPEED_BANDS))
    offset = (len(SPEED_PANEL_LIGHTS) - 1) / 2
    shown = bands[bands["light_condition"].isin(SPEED_PANEL_LIGHTS)]
    for i, light in enumerate(SPEED_PANEL_LIGHTS):
        rows = bands[bands["light_condition"] == light].set_index("speed_band").reindex(SPEED_BANDS)
        pos = xb + (i - offset) * width
        right.bar(pos, rows["pct_fatal"], width=width, color=LIGHT_COLOURS[light],
                  edgecolor="black", linewidth=0.5, label=SHORT_LABELS[light].replace("\n", " "),
                  hatch=None)
        # Hatch bars based on few crashes: their rate is unreliable.
        for p, r in zip(pos, rows.itertuples()):
            if r.low_n:
                right.bar(p, r.pct_fatal, width=width, color="none",
                          edgecolor="grey", hatch="///", linewidth=0)
        right.errorbar(pos, rows["pct_fatal"],
                       yerr=[rows["pct_fatal"] - rows["fatal_ci_low"],
                             rows["fatal_ci_high"] - rows["pct_fatal"]],
                       fmt="none", ecolor="dimgrey", capsize=2, linewidth=0.8)
    right.set_xticks(xb)
    right.set_xticklabels(SPEED_BANDS)
    label_axes(right, "Fatal share by light condition within speed zones",
               "Speed zone (posted limit, km/h)", "Fatal crashes (% of crashes)")
    handles, labels = right.get_legend_handles_labels()
    if shown["low_n"].any():
        handles.append(Patch(facecolor="white", edgecolor="grey", hatch="///"))
        labels.append(f"Fewer than {MIN_CRASHES} crashes (unreliable)")
    right.legend(handles, labels, loc="upper left", fontsize=9, title="Light condition")
    right.set_ylim(0, shown["fatal_ci_high"].max() * 1.15)

    fig.text(0.01, -0.02, "Injury crashes, Jan 2012 – Aug 2025. Unknown light "
             "condition ('Unk.', 'Dark Street lights unknown') excluded. "
             "Error bars: 95% confidence intervals. Right panel: 'Dark, lights off' "
             "omitted (only 426 crashes) and unknown speed zones excluded.",
             fontsize=9, ha="left")
    return save_fig(fig, "inv4_light_condition")


def print_key_numbers(df, table, bands):
    t = table.set_index("light_condition")
    print("\n  Key numbers:")
    print(f"  - {len(df):,} injury crashes with known light condition, Jan 2012 – Aug 2025")

    print("\n  Per light condition (% fatal [95% CI], deaths per 100 crashes, "
          "fatal rate vs day):")
    for light, r in t.iterrows():
        print(f"    {light:<24} n={int(r.crashes):>7,}  fatal {r.pct_fatal:4.2f}% "
              f"[{r.fatal_ci_low:.2f}–{r.fatal_ci_high:.2f}]  "
              f"deaths/100 {r.deaths_per_100_crashes:4.2f}  "
              f"x{r.fatal_rate_ratio_vs_day:.2f} [{r.ratio_ci_low:.2f}–{r.ratio_ci_high:.2f}]")

    u, lit = t.loc[UNLIT], t.loc[LIT]
    print(f"\n  - Dark with no street lights: {u.pct_fatal:.2f}% of crashes fatal vs "
          f"{t.loc[REFERENCE, 'pct_fatal']:.2f}% in daylight - "
          f"{u.fatal_rate_ratio_vs_day:.1f}x as likely to be fatal "
          f"(95% CI {u.ratio_ci_low:.1f}–{u.ratio_ci_high:.1f})")
    ratio, lo, hi = rate_ratio(u.fatal, u.crashes, lit.fatal, lit.crashes)
    print(f"  - Dark with no street lights vs dark with street lights on: "
          f"{u.pct_fatal:.2f}% vs {lit.pct_fatal:.2f}% fatal - {ratio:.1f}x "
          f"(95% CI {lo:.1f}–{hi:.1f})")
    print(f"  - Unlit crashes: {u.pct_of_crashes:.1f}% of crashes but "
          f"{u.pct_of_deaths:.1f}% of deaths")

    print("\n  Within speed zones (dark, no street lights vs day, % fatal):")
    b = bands.set_index(["speed_band", "light_condition"])
    for band in SPEED_BANDS:
        un, day = b.loc[(band, UNLIT)], b.loc[(band, REFERENCE)]
        ratio, lo, hi = rate_ratio(un.fatal, un.crashes, day.fatal, day.crashes)
        print(f"    {band:<11} {un.pct_fatal:.2f}% (n={int(un.crashes):,}) vs "
              f"{day.pct_fatal:.2f}% (n={int(day.crashes):,}) - {ratio:.1f}x "
              f"(95% CI {lo:.1f}–{hi:.1f})")
    print("    -> the gap narrows at high speed but remains, so speed zone explains "
          "only part of it")

    observed = pd.crosstab(df["LIGHT_CONDITION"], df["SEVERITY"] == FATAL)
    chi2, p, dof, _ = chi2_contingency(observed)
    v = np.sqrt(chi2 / (observed.values.sum() * (min(observed.shape) - 1)))
    print(f"\n  - Chi-square (light condition x fatal): chi2({dof}) = {chi2:,.1f}, "
          f"p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}, Cramér's V = {v:.3f}")
    print("    Note: with this many crashes almost any difference is 'significant';"
          " use the rate ratios and Cramér's V to judge the size.")
    print("  - Caveat: association, not cause. Unlit roads also differ in road type, "
          "distance to emergency care and driver behaviour (speed, alcohol and "
          "fatigue at night).")


def run(df):
    """Run investigation 4. Returns the list of saved file paths."""
    data = preprocess(df)
    table = summarise(data)
    bands = by_speed_band(data)
    paths = [plot(table, bands), save_table(table, "inv4"),
             save_table(bands, "inv4_by_speed")]
    print_key_numbers(data, table, bands)
    return paths
