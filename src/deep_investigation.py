"""Deep investigation: does the speed-zone effect on severity hold after
controlling for other factors? (extends investigation 1)

Investigation 1 found that crashes in higher speed zones are more often fatal
or serious (FSI). But high-speed roads are also rural, unlit, and have
different kinds of crashes (fewer pedestrians, more trees and poles), so part
of that pattern could come from those factors instead. Two approaches:

A. Grouped (stratified) comparisons - compare speed zones *within* groups of a
   possible confounder (e.g. only head-on crashes, only unlit roads), and pool
   the within-group odds ratios with the Mantel-Haenszel method.
B. Multivariable logistic regression - a sequence of models that add the
   possible confounders step by step and track how the speed-zone odds ratios
   change.

Outputs:
  figures/deep_01_stratified_gradient.png - FSI % by speed band within groups
  figures/deep_02_model_attenuation.png   - speed odds ratios across models M0-M3
  figures/deep_03_stratum_or.png          - 100 vs 60 km/h odds ratio per group
  results/deep_stratified.csv             - within-group odds ratios + pooled
  results/deep_models.csv                 - speed-zone odds ratios per model
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import inv6_severity_model as inv6
from src.inv1_speed_zone import MIN_CRASHES, wilson_ci
from src.inv4_light_condition import LIGHT_ORDER, SHORT_LABELS
from src.utils import COLOURS, label_axes, save_fig, save_table

# Predictors: column -> (display name, baseline category).
PREDICTORS = {
    "SPEED_ZONE": ("Speed zone", "60 km/h"),
    "LIGHT_CONDITION": ("Light condition", "Day"),
    "ROAD_GEOMETRY": ("Road geometry", "Not at intersection"),
    "ACCIDENT_TYPE": ("Crash type", "Collision with vehicle"),
    "RMA": ("Road class", "Local Road"),
    "DEG_URBAN_NAME": ("Urban/rural area", "MELB_URBAN"),
    "DIVIDED": ("Divided road", "Undivided"),
}
# Models add possible confounders step by step.
MODELS = [
    ("M0: speed only", ["SPEED_ZONE"]),
    ("M1: + light, geometry", ["SPEED_ZONE", "LIGHT_CONDITION", "ROAD_GEOMETRY"]),
    ("M2: + crash type", ["SPEED_ZONE", "LIGHT_CONDITION", "ROAD_GEOMETRY",
                          "ACCIDENT_TYPE"]),
    ("M3: + road class, urban/rural, divided",
     ["SPEED_ZONE", "LIGHT_CONDITION", "ROAD_GEOMETRY", "ACCIDENT_TYPE",
      "RMA", "DEG_URBAN_NAME", "DIVIDED"]),
]
MODEL_COLOURS = [COLOURS[7], COLOURS[0], COLOURS[1], COLOURS[3]]
MAIN_ROAD_CLASSES = ["Local Road", "Arterial Other", "Arterial Highway", "Freeway"]

SPEED_BINS = [0, 50, 60, 80, 200]
SPEED_BANDS = ["≤50", "60", "70–80", "≥90"]
HIGH, LOW = 100, 60  # speed zones compared within each group

STRATIFIERS = {
    # column: display name (levels chosen in stratifier_levels)
    "ACCIDENT_TYPE": "Crash type",
    "LIGHT_CONDITION": "Light condition",
    "RMA": "Road class",
    "STAT_DIV_NAME": "Metro vs country",
}
TOP_CRASH_TYPES = 6
RANDOM_STATE = 42


# ------------------------------------------------------------ preprocessing ---

def preprocess(df):
    """Same selection as investigation 6, plus a few derived columns."""
    # Steps 1-5 (incomplete period, non-injury, unknown speed zone, unknown
    # light/geometry, rare categories) are shared with investigation 6 so the
    # results are directly comparable. It prints each step.
    print("  Base selection (shared with investigation 6):")
    data = inv6.preprocess(df)

    # Step 6 - Tidy crash type labels: one category is stored in lower case
    # ("collision with some other object"), which would look like a
    # different category in tables and charts.
    data["ACCIDENT_TYPE"] = data["ACCIDENT_TYPE"].str[0].str.upper() + data["ACCIDENT_TYPE"].str[1:]

    # Step 7 - Group speed zones into 4 bands for the within-group comparisons.
    # Single zones (e.g. 90 km/h, 493 crashes) get too small once split by a
    # second variable; bands keep every cell large enough to compare.
    data["SPEED_BAND"] = pd.cut(data["SPEED_ZONE_KMH"], SPEED_BINS, labels=SPEED_BANDS)
    print(f"  Step 6-7: tidied crash type labels; added speed bands {SPEED_BANDS}")
    return data


def m3_subset(data):
    """Rows usable for model M3 (needs road class, urban/rural, divided)."""
    # M3 adds road-type variables. They are missing for some crashes (road
    # class/divided unknown for ~3.6%, urban/rural for ~0.5%), and road class
    # also has two tiny categories ('Non Arterial' 66, 'Proposed Road' 1)
    # that can't be estimated. Only M3 drops these rows, so M0-M2 keep the
    # full sample; M3's sample size is printed and compared.
    keep = (data["RMA"].isin(MAIN_ROAD_CLASSES) & data["DEG_URBAN_NAME"].notna()
            & data["DIVIDED"].notna())
    print(f"  Step 8 (M3 only): removed {int((~keep).sum()):,} crashes with unknown "
          f"or rare road class, urban/rural area or divided status "
          f"(road class unknown: {int(data['RMA'].isna().sum()):,})")
    return data[keep]


# ------------------------------------------------- A. grouped comparisons ---

def stratifier_levels(data, col):
    if col == "ACCIDENT_TYPE":
        return data[col].value_counts().head(TOP_CRASH_TYPES).index.tolist()
    if col == "LIGHT_CONDITION":
        return [l for l in LIGHT_ORDER if l in set(data[col])]
    if col == "RMA":
        return MAIN_ROAD_CLASSES
    return ["Metro", "Country"]


def level_label(col, level):
    if col == "LIGHT_CONDITION":
        return SHORT_LABELS.get(level, level).replace("\n", " ")
    return level


def gradient_table(data):
    """FSI % by speed band within each level of each stratifier."""
    rows = []
    for col, name in STRATIFIERS.items():
        for level in stratifier_levels(data, col):
            sub = data[data[col] == level]
            for band in SPEED_BANDS:
                cell = sub[sub["SPEED_BAND"] == band]
                n, k = len(cell), int(cell["FSI"].sum())
                lo, hi = wilson_ci(k, n) if n else (np.nan, np.nan)
                rows.append({"stratifier": name, "column": col, "level": level,
                             "speed_band": band, "crashes": n, "fsi": k,
                             "pct_fsi": k / n * 100 if n else np.nan,
                             "ci_low": lo, "ci_high": hi,
                             "low_n": n < MIN_CRASHES})
    return pd.DataFrame(rows)


def two_by_two(sub):
    """Counts for HIGH vs LOW km/h: a/b = FSI/non-FSI at HIGH, c/d at LOW."""
    hi = sub[sub["SPEED_ZONE_KMH"] == HIGH]["FSI"]
    lo = sub[sub["SPEED_ZONE_KMH"] == LOW]["FSI"]
    return hi.sum(), (1 - hi).sum(), lo.sum(), (1 - lo).sum()


def mantel_haenszel(tables):
    """Pooled odds ratio across strata with a 95% CI (Robins-Breslow-
    Greenland variance). Each stratum is compared only with itself, so the
    pooled value is 'the speed effect with this variable held constant'."""
    a, b, c, d = (np.array(x, dtype=float) for x in zip(*tables))
    n = a + b + c + d
    R, S = a * d / n, b * c / n
    P, Q = (a + d) / n, (b + c) / n
    or_mh = R.sum() / S.sum()
    var = ((P * R).sum() / (2 * R.sum() ** 2)
           + (P * S + Q * R).sum() / (2 * R.sum() * S.sum())
           + (Q * S).sum() / (2 * S.sum() ** 2))
    se = np.sqrt(var)
    return or_mh, or_mh * np.exp(-1.96 * se), or_mh * np.exp(1.96 * se)


def stratum_odds_ratios(data):
    """100 vs 60 km/h odds ratio within each stratum + pooled per stratifier."""
    rows = []
    for col, name in STRATIFIERS.items():
        tables = []
        for level in stratifier_levels(data, col):
            a, b, c, d = two_by_two(data[data[col] == level])
            row = {"stratifier": name, "column": col, "level": level,
                   "fsi_100": int(a), "non_fsi_100": int(b),
                   "fsi_60": int(c), "non_fsi_60": int(d), "pooled": False}
            # Skip strata with an empty cell (odds ratio undefined) or so few
            # crashes at 100 km/h that the estimate is meaningless.
            if min(a, b, c, d) == 0 or a + b < MIN_CRASHES:
                row.update(odds_ratio=np.nan, ci_low=np.nan, ci_high=np.nan,
                           skipped=True)
            else:
                orr = (a * d) / (b * c)
                se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
                row.update(odds_ratio=orr, ci_low=orr * np.exp(-1.96 * se),
                           ci_high=orr * np.exp(1.96 * se), skipped=False)
                tables.append((a, b, c, d))
            rows.append(row)
        or_mh, lo, hi = mantel_haenszel(tables)
        rows.append({"stratifier": name, "column": col, "level": "Pooled (Mantel-Haenszel)",
                     "odds_ratio": or_mh, "ci_low": lo, "ci_high": hi,
                     "pooled": True, "skipped": False})
    return pd.DataFrame(rows)


# ------------------------------------------- B. multivariable regression ---

def fit_model(data, columns):
    """Fit one logistic regression; return speed-zone rows + AUC."""
    X, meta = inv6.design_matrix(data, {c: PREDICTORS[c] for c in columns})
    y = data["FSI"].values
    model = LogisticRegression(penalty=None, max_iter=3000).fit(X, y)
    coefs = meta.copy()
    coefs["coef"] = model.coef_[0]
    coefs["se"] = inv6.standard_errors(model, X)
    coefs["odds_ratio"] = np.exp(coefs["coef"])
    coefs["ci_low"] = np.exp(coefs["coef"] - 1.96 * coefs["se"])
    coefs["ci_high"] = np.exp(coefs["coef"] + 1.96 * coefs["se"])
    coefs["p_value"] = 2 * norm.sf(np.abs(coefs["coef"] / coefs["se"]))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc = cross_val_score(LogisticRegression(penalty=None, max_iter=3000), X, y,
                          cv=folds, scoring="roc_auc").mean()
    return coefs, auc


def fit_models(data, data_m3):
    rows, aucs, all_coefs = [], {}, {}
    for name, columns in MODELS:
        sample = data_m3 if "RMA" in columns else data
        coefs, auc = fit_model(sample, columns)
        aucs[name] = auc
        all_coefs[name] = coefs
        speed = coefs[coefs["column"] == "SPEED_ZONE"].copy()
        speed.insert(0, "model", name)
        speed["rows_used"] = len(sample)
        speed["cv_auc"] = auc
        rows.append(speed)
    table = pd.concat(rows, ignore_index=True)
    table = table[["model", "category", "n", "odds_ratio", "ci_low", "ci_high",
                   "p_value", "rows_used", "cv_auc"]].rename(columns={"category": "speed_zone"})
    return table, aucs, all_coefs


def per_10_kmh(data_m3):
    """M3 with speed as a straight-line number: odds ratio per +10 km/h."""
    others = {c: PREDICTORS[c] for c in MODELS[-1][1] if c != "SPEED_ZONE"}
    X, _ = inv6.design_matrix(data_m3, others)
    X.insert(0, "speed_per_10kmh", data_m3["SPEED_ZONE_KMH"].values / 10)
    model = LogisticRegression(penalty=None, max_iter=3000).fit(X, data_m3["FSI"].values)
    coef = model.coef_[0][0]
    se = inv6.standard_errors(model, X)[0]
    return np.exp(coef), np.exp(coef - 1.96 * se), np.exp(coef + 1.96 * se)


# ------------------------------------------------------------------ plots ---

def plot_gradient(grad):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey=True)
    x = np.arange(len(SPEED_BANDS))
    for ax, (col, name) in zip(axes.flat, STRATIFIERS.items()):
        sub = grad[grad["column"] == col]
        for i, level in enumerate(sub["level"].unique()):
            g = sub[sub["level"] == level].set_index("speed_band").reindex(SPEED_BANDS)
            # Hide points based on too few crashes.
            y = g["pct_fsi"].where(~g["low_n"])
            lo = (g["pct_fsi"] - g["ci_low"]).where(~g["low_n"])
            hi = (g["ci_high"] - g["pct_fsi"]).where(~g["low_n"])
            offset = (i - 2.5) * 0.04
            ax.errorbar(x + offset, y, yerr=[lo, hi], marker="o", markersize=5,
                        capsize=2, linewidth=1.6, color=COLOURS[i % len(COLOURS)],
                        label=level_label(col, level))
        ax.set_xticks(x)
        ax.set_xticklabels(SPEED_BANDS)
        label_axes(ax, f"Within {name.lower()}", "Speed zone (km/h band)",
                   "Fatal or serious (% of crashes)")
        ax.legend(fontsize=8, loc="upper left", title=name, title_fontsize=9)
    axes[0, 0].set_ylim(0, 85)
    fig.suptitle("Does severity still rise with speed zone inside each group?",
                 fontsize=15, fontweight="bold")
    fig.text(0.01, -0.01, "Injury crashes, Jan 2012 – Aug 2025. Error bars: 95% CI. "
             f"Points based on fewer than {MIN_CRASHES} crashes are hidden. "
             f"Crash type: {TOP_CRASH_TYPES} most common types.", fontsize=9, ha="left")
    return save_fig(fig, "deep_01_stratified_gradient")


def plot_attenuation(models):
    zones = [z for z in models["speed_zone"].unique()]
    zones = sorted(zones, key=lambda s: int(s.split()[0]))
    names = [m for m, _ in MODELS]
    fig, ax = plt.subplots(figsize=(11, 8))
    step = 0.18
    for j, (name, colour) in enumerate(zip(names, MODEL_COLOURS)):
        m = models[models["model"] == name].set_index("speed_zone").reindex(zones)
        y = np.arange(len(zones))[::-1] + (1.5 - j) * step
        ax.errorbar(m["odds_ratio"], y,
                    xerr=[m["odds_ratio"] - m["ci_low"], m["ci_high"] - m["odds_ratio"]],
                    fmt="o", color=colour, markersize=5, capsize=2, linewidth=1,
                    label=name)
    ax.axvline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ticks = [0.6, 0.8, 1, 1.25, 1.5, 2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(np.arange(len(zones))[::-1])
    ax.set_yticklabels(zones)
    ax.grid(axis="y", visible=False)
    label_axes(ax, "Speed-zone odds ratios as possible confounders are added",
               "Odds ratio of a fatal or serious outcome vs 60 km/h (log scale)",
               "Speed zone (posted limit)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="black", linestyle="--"))
    labels.append("No difference from 60 km/h")
    # Top right is empty: low speed zones all sit left of 1.
    ax.legend(handles, labels, loc="upper right", fontsize=9, title="Model")
    fig.text(0.01, -0.02, "M0–M2: 177,200 crashes; M3: crashes with known road class, "
             "urban/rural area and divided status. Error bars: 95% CI.",
             fontsize=9, ha="left")
    return save_fig(fig, "deep_02_model_attenuation")


def plot_stratum_or(strata):
    shown = strata[~strata["skipped"]]
    fig, ax = plt.subplots(figsize=(11, 10))
    y, ypos, labels = 0, [], []
    for col, name in STRATIFIERS.items():
        sub = shown[shown["column"] == col]
        for r in sub.itertuples():
            ypos.append(y)
            labels.append(f"{name}: pooled" if r.pooled else level_label(col, r.level))
            y -= 1
        y -= 0.7
    ypos = np.array(ypos)
    pooled = shown["pooled"].values
    colours = [COLOURS[list(STRATIFIERS).index(c)] for c in shown["column"]]
    ax.errorbar(shown["odds_ratio"], ypos,
                xerr=[shown["odds_ratio"] - shown["ci_low"],
                      shown["ci_high"] - shown["odds_ratio"]],
                fmt="none", ecolor="grey", capsize=2, linewidth=1)
    ax.scatter(shown["odds_ratio"][~pooled], ypos[~pooled],
               c=np.array(colours)[~pooled], s=35, zorder=3)
    ax.scatter(shown["odds_ratio"][pooled], ypos[pooled],
               c=np.array(colours)[pooled], s=110, marker="D", edgecolors="black", zorder=4)
    ax.axvline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ticks = [0.5, 1, 1.5, 2, 3, 5]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.grid(axis="y", visible=False)
    label_axes(ax, f"{HIGH} vs {LOW} km/h within each group",
               f"Odds ratio of a fatal or serious outcome, {HIGH} vs {LOW} km/h (log scale)",
               "Group")
    handles = [Line2D([], [], marker="o", linestyle="", color=COLOURS[i], label=n)
               for i, n in enumerate(STRATIFIERS.values())]
    handles += [Line2D([], [], marker="D", linestyle="", color="white",
                       markeredgecolor="black", markersize=8,
                       label="Pooled (Mantel–Haenszel)"),
                Line2D([], [], color="grey", label="95% confidence interval"),
                Line2D([], [], color="black", linestyle="--", label="No difference")]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    skipped = strata[strata["skipped"]]
    note = "Error bars: 95% CI (log method)."
    if len(skipped):
        note += " Not shown (too few 100 km/h crashes or an empty cell): " + ", ".join(
            f"{level_label(r.column, r.level)}" for r in skipped.itertuples()) + "."
    fig.text(0.01, -0.02, note, fontsize=9, ha="left", wrap=True)
    return save_fig(fig, "deep_03_stratum_or")


# ---------------------------------------------------------------- summary ---

def print_summary(data, data_m3, models, aucs, strata, per10):
    m = models.set_index(["model", "speed_zone"])
    first, last = MODELS[0][0], MODELS[-1][0]

    print("\n  Key numbers:")
    print(f"  - {len(data):,} crashes (M0–M2), {len(data_m3):,} for M3; "
          f"{data['FSI'].mean():.1%} fatal or serious")
    print("  - Speed-zone odds ratios vs 60 km/h, unadjusted (M0) -> fully adjusted (M3):")
    for zone in ["40 km/h", "70 km/h", "80 km/h", "100 km/h", "110 km/h"]:
        a, b = m.loc[(first, zone)], m.loc[(last, zone)]
        print(f"      {zone:>8}: {a.odds_ratio:.2f} [{a.ci_low:.2f}–{a.ci_high:.2f}] -> "
              f"{b.odds_ratio:.2f} [{b.ci_low:.2f}–{b.ci_high:.2f}]")
    print("  - 100 km/h across models: " + " -> ".join(
        f"{name.split(':')[0]} {m.loc[(name, '100 km/h')].odds_ratio:.2f}" for name, _ in MODELS))
    raw, adj = m.loc[(first, "100 km/h")].odds_ratio, m.loc[(last, "100 km/h")].odds_ratio
    shrink = (1 - (adj - 1) / (raw - 1)) * 100
    print(f"    The extra odds at 100 km/h shrink from +{(raw - 1) * 100:.0f}% to "
          f"+{(adj - 1) * 100:.0f}% ({shrink:.0f}% of the raw effect explained by the "
          "added factors)")
    print(f"  - Speed as a number (M3): each +10 km/h in the posted limit multiplies the "
          f"odds by {per10[0]:.3f} [95% CI {per10[1]:.3f}–{per10[2]:.3f}]")
    print("  - Model fit (cross-validated AUC): " + ", ".join(
        f"{name.split(':')[0]} {auc:.3f}" for name, auc in aucs.items()))

    groups = strata[~strata["pooled"] & ~strata["skipped"]]
    above = int((groups["ci_low"] > 1).sum())
    print(f"\n  - Within-group comparison ({HIGH} vs {LOW} km/h): odds ratio > 1 in "
          f"{int((groups['odds_ratio'] > 1).sum())} of {len(groups)} groups; the 95% CI "
          f"is entirely above 1 in {above}")
    for r in strata[strata["pooled"]].itertuples():
        print(f"      pooled within {r.stratifier.lower():<17} OR {r.odds_ratio:.2f} "
              f"[{r.ci_low:.2f}–{r.ci_high:.2f}]")
    weakest = groups.loc[groups["odds_ratio"].idxmin()]
    strongest = groups.loc[groups["odds_ratio"].idxmax()]
    print(f"      weakest group: {level_label(weakest.column, weakest.level)} "
          f"(OR {weakest.odds_ratio:.2f} [{weakest.ci_low:.2f}–{weakest.ci_high:.2f}]); "
          f"strongest: {level_label(strongest.column, strongest.level)} "
          f"(OR {strongest.odds_ratio:.2f})")
    skipped = strata[strata["skipped"]]
    if len(skipped):
        print("      skipped (too few 100 km/h crashes): " + ", ".join(
            level_label(r.column, r.level) for r in skipped.itertuples()))

    # How crash type differs by speed zone (the main confounder).
    high = data["SPEED_ZONE_KMH"] >= 90
    mix = pd.crosstab(data["ACCIDENT_TYPE"], high, normalize="columns") * 100
    print("  - Crash mix, zones <90 vs >=90 km/h: "
          f"struck pedestrian {mix.loc['Struck Pedestrian', False]:.1f}% vs "
          f"{mix.loc['Struck Pedestrian', True]:.1f}%; fixed object "
          f"{mix.loc['Collision with a fixed object', False]:.1f}% vs "
          f"{mix.loc['Collision with a fixed object', True]:.1f}%")

    print("\n  Findings:")
    print(f"  1. The speed-zone effect holds. After adding light, geometry, crash type, "
          f"road class, urban/rural area and divided roads, 100 km/h zones still have "
          f"{(adj - 1) * 100:.0f}% higher odds of a fatal/serious outcome than 60 km/h "
          f"(OR {adj:.2f}), and the pattern rises steadily with the limit "
          f"({per10[0]:.2f}x odds per 10 km/h).")
    print(f"  2. Part of the raw effect ({shrink:.0f}%) is explained by the other factors. "
          "Crash type matters most: high-speed zones have many more fixed-object and "
          "rollover crashes and far fewer pedestrian crashes. Adjusting for it lowers "
          "the 40-50 km/h odds ratios (pedestrian crashes were masking their advantage).")
    print(f"  3. The effect appears inside almost every group tested ({above} of "
          f"{len(groups)} within-group CIs are entirely above 1), in metro and country "
          "areas and on every road class, so no single confounder explains it.")
    # 110 vs 100 km/h: does road design explain the lower 110 km/h odds?
    m1_name = MODELS[1][0]
    m1_100, m1_110 = m.loc[(m1_name, "100 km/h")], m.loc[(m1_name, "110 km/h")]
    m3_100, m3_110 = m.loc[(last, "100 km/h")], m.loc[(last, "110 km/h")]
    overlap = m3_110.ci_low <= m3_100.ci_high and m3_100.ci_low <= m3_110.ci_high
    # Road design at 100 vs 110 km/h (crashes with known road class/divided).
    design = {}
    for kmh in (100, 110):
        z = data[data["SPEED_ZONE_KMH"] == kmh]
        design[kmh] = (z["RMA"].dropna().eq("Freeway").mean() * 100,
                       z["DIVIDED"].dropna().eq("Divided").mean() * 100)
    print(f"  - Road design: freeway {design[110][0]:.0f}% / divided {design[110][1]:.0f}% "
          f"of 110 km/h crashes vs freeway {design[100][0]:.0f}% / divided "
          f"{design[100][1]:.0f}% of 100 km/h crashes")
    print(f"  4. 110 km/h zones: before adjusting for road type (M1) their odds ratio "
          f"({m1_110.odds_ratio:.2f}) was below 100 km/h ({m1_100.odds_ratio:.2f}); "
          f"after adding road class and divided roads (M3) it is "
          f"{m3_110.odds_ratio:.2f} vs {m3_100.odds_ratio:.2f}"
          f"{' (CIs overlap - no clear difference)' if overlap else ''}. 110 km/h roads "
          "are mostly divided freeways, so their lower raw severity reflects road design "
          "(barriers, no oncoming traffic) rather than the higher limit being safer.")

    print("\n  Possible confounders and limits:")
    print("  - Rural roads: high-speed zones are mostly rural. M3 adjusts for road class and "
          "urban/rural area, but other rural factors remain (distance to hospital and "
          "ambulance response times, older vehicles, unsealed shoulders, trees close to "
          "the road).")
    print("  - No traffic volume (exposure) data: we can compare how *severe* crashes are, "
          "not how *likely* a crash is per km travelled. Nothing here says 100 km/h roads "
          "have more crashes per trip.")
    print("  - Only injury crashes are recorded: if minor crashes on high-speed rural roads "
          "are less often reported than in the city, the FSI share there is overstated.")
    print("  - Posted limit is not actual travel speed: congestion and speeding are not "
          "measured.")
    print("  - Crash type is partly *caused* by speed (faster vehicles are more likely to "
          "leave the road and hit a fixed object), so adjusting for it removes some of "
          "the real speed effect. M2/M3 are therefore conservative estimates.")
    print("  - Not in the data: driver age and experience, alcohol/drugs, fatigue, "
          "seatbelt/helmet use, vehicle type and safety rating.")

    print("\n  Confidence level:")
    print("  - HIGH that higher speed zones are associated with more severe crashes: the "
          "effect is large, tightly estimated, rises steadily with the limit, and survives "
          "every adjustment and every group comparison.")
    print(f"  - MODERATE on the size: adjusted OR about {adj:.1f} for 100 vs 60 km/h, but "
          f"it ranges from {m.xs('100 km/h', level='speed_zone')['odds_ratio'].min():.2f} to "
          f"{m.xs('100 km/h', level='speed_zone')['odds_ratio'].max():.2f} depending on "
          "the model, and adjusting for crash type may remove part of the true effect.")
    print("  - LOW that speed limits alone *cause* the difference: this is observational "
          "data with unmeasured confounders (exposure, actual speed, emergency response).")


def run(df):
    """Run the deep investigation. Returns the list of saved file paths."""
    data = preprocess(df)
    data_m3 = m3_subset(data)

    grad = gradient_table(data)
    strata = stratum_odds_ratios(data)
    models, aucs, _ = fit_models(data, data_m3)
    per10 = per_10_kmh(data_m3)

    strata_out = pd.concat([strata, grad.assign(pooled=False, skipped=False)],
                           keys=["odds_ratio_100_vs_60", "fsi_by_speed_band"],
                           names=["table", None]).reset_index(level=0)
    paths = [plot_gradient(grad), plot_attenuation(models), plot_stratum_or(strata),
             save_table(strata_out.round(4), "deep_stratified"),
             save_table(models.round(4), "deep_models")]
    print_summary(data, data_m3, models, aucs, strata, per10)
    return paths
