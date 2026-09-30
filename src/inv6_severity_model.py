"""Investigation 6: What predicts a fatal or serious crash? (statistical tests)

1. Chi-square test of independence: is speed zone associated with whether a
   crash is fatal/serious (FSI)? Effect size: Cramér's V.
2. Logistic regression: FSI (yes/no) ~ speed zone + light condition + road
   geometry, so each factor's effect is estimated with the other two held the
   same. Reported as odds ratios with 95% confidence intervals and p-values.

Outputs:
  figures/inv6_severity_model.png - stacked bar per speed zone + odds ratio plot
  results/inv6.csv                - % fatal / serious per speed zone (model data)
  results/inv6_logit.csv          - logistic regression coefficients
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency, norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import inv1_speed_zone
from src.inv1_speed_zone import FATAL, MIN_CRASHES, NON_INJURY, SERIOUS
from src.inv4_light_condition import LIGHT_ORDER, SHORT_LABELS
from src.utils import COLOURS, SEVERITY_COLOURS, label_axes, save_fig, save_table

# Baseline (reference) category for each predictor: the most common value, so
# every other category is compared with a typical crash.
PREDICTORS = {
    # column: (display name, baseline)
    "SPEED_ZONE": ("Speed zone", "60 km/h"),
    "LIGHT_CONDITION": ("Light condition", "Day"),
    "ROAD_GEOMETRY": ("Road geometry", "Not at intersection"),
}
PREDICTOR_COLOURS = dict(zip(PREDICTORS, [COLOURS[0], COLOURS[3], COLOURS[2]]))
RANDOM_STATE = 42


def preprocess(df):
    """Select the crashes used for the tests and the model, printing each step."""
    n_start = len(df)
    print(f"  Starting rows: {n_start:,}")

    # Step 1 - Drop the incomplete period (Sep 2025 onwards): severe crashes
    # are entered first, so these months would inflate the FSI share.
    incomplete = df["INCOMPLETE_PERIOD"]
    print(f"  Step 1: removed {int(incomplete.sum()):,} crashes from the incomplete period")
    df = df[~incomplete]

    # Step 2 - Remove non-injury crashes: the dataset only covers injury
    # crashes, so these few records are out of scope.
    non_injury = df["SEVERITY"] == NON_INJURY
    print(f"  Step 2: removed {int(non_injury.sum()):,} non-injury crashes")
    df = df[~non_injury]

    # Step 3 - Remove unknown speed zones ("Not known", "Camping grounds or
    # off road", "Other speed limit"). They have no km/h value, and a
    # regression can't use a missing predictor.
    unknown_speed = df["SPEED_ZONE_KMH"].isna()
    print(f"  Step 3: removed {int(unknown_speed.sum()):,} crashes with unknown speed zone")
    df = df[~unknown_speed]

    # Step 4 - Remove unknown light condition / road geometry. Every predictor
    # must be known for each crash; guessing values would bias the estimates.
    unknown_other = df["LIGHT_CONDITION"].isna() | df["ROAD_GEOMETRY"].isna()
    print(f"  Step 4: removed {int(unknown_other.sum()):,} crashes with unknown light "
          f"condition ({int(df['LIGHT_CONDITION'].isna().sum()):,}) or road geometry "
          f"({int(df['ROAD_GEOMETRY'].isna().sum()):,})")
    df = df[~unknown_other].copy()

    # Step 5 - Remove rare categories (< MIN_CRASHES crashes). With a handful
    # of crashes their coefficients are mostly noise and can destabilise the
    # fit (e.g. 75 km/h has 20 crashes, 'Road closure' 6).
    df["SPEED_ZONE"] = df["SPEED_ZONE_KMH"].astype(int).astype(str) + " km/h"
    rare = pd.Series(False, index=df.index)
    for col in PREDICTORS:
        counts = df[col].value_counts()
        small = counts[counts < MIN_CRASHES]
        if len(small):
            print(f"  Step 5: {PREDICTORS[col][0]}: removed rare categories "
                  + ", ".join(f"'{k}' ({v})" for k, v in small.items()))
        rare |= df[col].isin(small.index)
    df = df[~rare].copy()

    df["FSI"] = df["SEVERITY"].isin([FATAL, SERIOUS]).astype(int)
    print(f"  Rows used: {len(df):,} ({len(df) / n_start:.1%} of all rows); "
          f"{df['FSI'].mean():.1%} fatal or serious")
    return df


def chi_square(df):
    """Speed zone x FSI (yes/no): chi-square test of independence + Cramér's V."""
    table = pd.crosstab(df["SPEED_ZONE_KMH"], df["FSI"])
    chi2, p, dof, _ = chi2_contingency(table)
    v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
    return chi2, p, dof, v


def design_matrix(df, predictors=None):
    """One 0/1 column per non-baseline category ('one-hot' coding).
    The baseline category gets no column: its effect is in the intercept, and
    every coefficient is a comparison against it.
    predictors: {column: (display name, baseline)}; defaults to PREDICTORS."""
    predictors = PREDICTORS if predictors is None else predictors
    columns, meta = {}, []
    for col, (name, baseline) in predictors.items():
        if col == "SPEED_ZONE":
            levels = sorted(df[col].unique(), key=lambda s: int(s.split()[0]))
        elif col == "LIGHT_CONDITION":
            levels = [l for l in LIGHT_ORDER if l in set(df[col])]
        else:
            levels = df[col].value_counts().index.tolist()
        for level in levels:
            if level == baseline:
                continue
            columns[f"{col}={level}"] = (df[col] == level).astype(float).values
            meta.append({"predictor": name, "column": col, "category": level,
                         "baseline": baseline, "n": int((df[col] == level).sum())})
    return pd.DataFrame(columns, index=df.index), pd.DataFrame(meta)


def standard_errors(model, X):
    """Standard errors for the coefficients of an unpenalised logistic regression.

    scikit-learn gives coefficients but not their uncertainty. The standard
    formula: the covariance of the estimates is the inverse of X' W X, where
    X includes an intercept column and W holds p(1 - p) for each crash (p =
    the model's predicted probability). Crashes the model is unsure about
    (p near 0.5) carry the most information."""
    Xc = np.column_stack([np.ones(len(X)), X.values])
    p = model.predict_proba(X)[:, 1]
    w = p * (1 - p)
    cov = np.linalg.inv((Xc * w[:, None]).T @ Xc)
    return np.sqrt(np.diag(cov))[1:]  # drop the intercept


def fit_logit(df):
    X, meta = design_matrix(df)
    y = df["FSI"].values
    # penalty=None: plain (unpenalised) maximum likelihood, so coefficients
    # can be interpreted directly and standard errors are valid.
    model = LogisticRegression(penalty=None, max_iter=2000).fit(X, y)

    coefs = meta.copy()
    coefs["coef"] = model.coef_[0]
    coefs["se"] = standard_errors(model, X)
    coefs["z"] = coefs["coef"] / coefs["se"]
    coefs["p_value"] = 2 * norm.sf(np.abs(coefs["z"]))
    coefs["odds_ratio"] = np.exp(coefs["coef"])
    coefs["or_ci_low"] = np.exp(coefs["coef"] - 1.96 * coefs["se"])
    coefs["or_ci_high"] = np.exp(coefs["coef"] + 1.96 * coefs["se"])

    # How well do these three factors predict severity? 5-fold cross-validated
    # AUC: 0.5 = coin flip, 1.0 = perfect.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc = cross_val_score(LogisticRegression(penalty=None, max_iter=2000), X, y,
                          cv=folds, scoring="roc_auc")
    return model, X, coefs, auc


def typical_probabilities(model, X, coefs):
    """Predicted FSI probability for a 'typical' crash (daylight, not at an
    intersection) in each speed zone: turns odds ratios into percentages."""
    base = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)
    probs = {"60 km/h": model.predict_proba(base)[0, 1]}
    for r in coefs[coefs["column"] == "SPEED_ZONE"].itertuples():
        row = base.copy()
        row[f"SPEED_ZONE={r.category}"] = 1.0
        probs[r.category] = model.predict_proba(row)[0, 1]
    return pd.Series(probs).sort_index(key=lambda i: i.str.split().str[0].astype(int))


def display_label(col, category):
    if col == "LIGHT_CONDITION":
        return SHORT_LABELS.get(category, category).replace("\n", " ")
    return category


def plot(summary, coefs):
    fig, (left, right) = plt.subplots(1, 2, figsize=(18, 8),
                                      gridspec_kw={"width_ratios": [1, 1.05]})

    # --- Left: % fatal + % serious per speed zone (same style as inv1).
    x = np.arange(len(summary))
    left.bar(x, summary["pct_fatal"], color=SEVERITY_COLOURS[FATAL], label="Fatal")
    left.bar(x, summary["pct_serious"], bottom=summary["pct_fatal"],
             color=SEVERITY_COLOURS[SERIOUS], label="Serious injury")
    left.errorbar(x, summary["pct_fsi"],
                  yerr=[summary["pct_fsi"] - summary["fsi_ci_low"],
                        summary["fsi_ci_high"] - summary["pct_fsi"]],
                  fmt="none", ecolor="black", capsize=3, linewidth=1,
                  label="95% CI (fatal + serious)")
    for xi, r in zip(x, summary.itertuples()):
        left.text(xi, r.fsi_ci_high + 0.8, f"{r.pct_fsi:.1f}%", ha="center",
                  va="bottom", fontsize=9, fontweight="bold")
    left.set_xticks(x)
    left.set_xticklabels([f"{s}\n(n={n:,})" for s, n in
                          zip(summary["speed_zone_kmh"], summary["crashes"])], fontsize=8)
    left.set_ylim(0, summary["fsi_ci_high"].max() + 8)
    label_axes(left, "Observed: fatal and serious crashes by speed zone",
               "Posted speed limit (km/h)", "Crashes (% of crashes in speed zone)")
    left.legend(loc="upper left", fontsize=9)

    # --- Right: forest plot of odds ratios, grouped by predictor.
    y, ypos, labels = 0, [], []
    for col in PREDICTORS:
        rows = coefs[coefs["column"] == col]
        for r in rows.itertuples():
            ypos.append(y)
            labels.append(display_label(col, r.category))
            y -= 1
        y -= 0.8  # gap between predictor groups
    ypos = np.array(ypos)
    colours = [PREDICTOR_COLOURS[c] for c in coefs["column"]]
    right.errorbar(coefs["odds_ratio"], ypos,
                   xerr=[coefs["odds_ratio"] - coefs["or_ci_low"],
                         coefs["or_ci_high"] - coefs["odds_ratio"]],
                   fmt="none", ecolor="grey", capsize=2, linewidth=1)
    right.scatter(coefs["odds_ratio"], ypos, c=colours, s=40, zorder=3)
    right.axvline(1, color="black", linestyle="--", linewidth=1)
    right.set_xscale("log")
    ticks = [0.5, 0.7, 1, 1.5, 2, 3]
    right.set_xticks(ticks)
    right.set_xticklabels([f"{t:g}" for t in ticks])
    right.minorticks_off()
    lo = min(0.5, coefs["or_ci_low"].min() * 0.9)
    hi = max(2.5, coefs["or_ci_high"].max() * 1.1)
    right.set_xlim(lo, hi)
    right.set_yticks(ypos)
    right.set_yticklabels(labels, fontsize=9)
    right.grid(axis="y", visible=False)
    label_axes(right, "Logistic regression: odds of a fatal or serious outcome",
               "Odds ratio (log scale; 1 = same as baseline)", "Category")
    handles = [Line2D([], [], marker="o", linestyle="", color=PREDICTOR_COLOURS[c],
                      label=f"{name} (vs {base.replace('Day', 'daylight').replace('Not', 'not')})")
               for c, (name, base) in PREDICTORS.items()]
    handles.append(Line2D([], [], color="grey", label="95% confidence interval"))
    handles.append(Line2D([], [], color="black", linestyle="--",
                          label="No difference from baseline"))
    # Top right is empty (the lowest speed zones sit left of 1).
    right.legend(handles=handles, loc="upper right", fontsize=9)

    fig.text(0.01, -0.02, "Injury crashes, Jan 2012 – Aug 2025, with known speed zone, "
             "light condition and road geometry. Odds ratios from one model, so each "
             "is adjusted for the other two factors.", fontsize=9, ha="left")
    return save_fig(fig, "inv6_severity_model")


def plain_english(r):
    """One sentence explaining an odds ratio."""
    base = r.baseline.replace("Day", "daylight")
    base = base[0].lower() + base[1:]
    cat = display_label(r.column, r.category)
    ci = f"95% CI {r.or_ci_low:.2f}–{r.or_ci_high:.2f}"
    if r.p_value >= 0.05:
        return (f"{cat}: odds ratio {r.odds_ratio:.2f} ({ci}, p = {r.p_value:.2f}) - "
                f"no clear difference from {base}")
    if r.odds_ratio >= 1:
        change = f"{(r.odds_ratio - 1) * 100:.0f}% higher odds"
    else:
        change = f"{(1 - r.odds_ratio) * 100:.0f}% lower odds"
    p = "p < 0.001" if r.p_value < 0.001 else f"p = {r.p_value:.3f}"
    return (f"{cat}: {change} of a fatal/serious outcome than {base} "
            f"(OR {r.odds_ratio:.2f}, {ci}, {p})")


def print_key_numbers(df, chi, coefs, auc, probs):
    chi2, p, dof, v = chi
    print("\n  Key numbers:")
    print(f"  - {len(df):,} injury crashes, Jan 2012 – Aug 2025; "
          f"{df['FSI'].mean():.1%} fatal or serious")

    print(f"\n  Chi-square test (speed zone x fatal/serious yes-no):")
    print(f"    chi2({dof}) = {chi2:,.1f}, p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}, "
          f"Cramér's V = {v:.3f}")
    print("    In plain English: if speed zone and severity were unrelated, a "
          "difference this large would almost never happen by chance (p < 0.001), "
          "so they are associated.")
    print(f"    Cramér's V = {v:.2f} (0 = no link, 1 = perfect link; ~0.1 = weak): the "
          "link is real but weak - speed zone on its own explains only a small part "
          "of which crashes are severe.")
    print(f"    With {len(df):,} crashes even small differences give tiny p-values, so "
          "judge importance by Cramér's V and the odds ratios, not p.")
    print("    (Investigation 1 tested speed zone x 3 severity levels: V = 0.104.)")

    print("\n  Logistic regression (fatal/serious vs speed zone + light + road geometry):")
    print("  Each odds ratio compares a category with its baseline while the other two "
          "factors are held the same.")
    for col, (name, base) in PREDICTORS.items():
        print(f"\n    {name} (baseline: {base}):")
        for r in coefs[coefs["column"] == col].itertuples():
            print(f"      - {plain_english(r)}")

    print("\n  Biggest effects:")
    for col, (name, _) in PREDICTORS.items():
        rows = coefs[coefs["column"] == col]
        top = rows.loc[(np.log(rows["odds_ratio"])).abs().idxmax()]
        print(f"    - {name}: {display_label(col, top.category)} "
              f"(OR {top.odds_ratio:.2f})")

    print(f"\n  Odds ratios vs probabilities: {df['FSI'].mean():.0%} of crashes are "
          "fatal/serious, so odds ratios exaggerate the change in probability.")
    print("  Predicted chance of a fatal/serious outcome for a typical crash "
          "(daylight, not at an intersection):")
    print("    " + ", ".join(f"{k}: {v:.0%}" for k, v in probs.items()))
    print(f"    e.g. 60 km/h {probs['60 km/h']:.0%} -> 100 km/h {probs['100 km/h']:.0%} "
          f"(odds ratio {coefs.set_index('category').loc['100 km/h', 'odds_ratio']:.2f}, "
          f"but the probability is {probs['100 km/h'] / probs['60 km/h']:.2f}x as high)")

    print(f"\n  - Predictive power: 5-fold cross-validated AUC = {auc.mean():.3f} "
          f"(range {auc.min():.3f}–{auc.max():.3f}).")
    print("    In plain English: given two crashes, one fatal/serious and one not, the "
          f"model ranks the severe one higher {auc.mean():.0%} of the time (50% = "
          "guessing). These three factors matter, but most of what makes a crash "
          "severe (vehicle, seatbelt use, actual speed, age, etc.) is not in them.")
    print("  - Caveats: association, not cause. The model only covers crashes that "
          "caused an injury, and road type, location and driver behaviour that "
          "go with each factor are not fully controlled for.")


def run(df):
    """Run investigation 6. Returns the list of saved file paths."""
    data = preprocess(df)
    summary = inv1_speed_zone.summarise(data)
    chi = chi_square(data)
    model, X, coefs, auc = fit_logit(data)
    probs = typical_probabilities(model, X, coefs)

    table = coefs[["predictor", "category", "baseline", "n", "coef", "se", "z",
                   "p_value", "odds_ratio", "or_ci_low", "or_ci_high"]].copy()
    table["p_value"] = table["p_value"].map(lambda p: float(f"{p:.3g}"))
    paths = [plot(summary, coefs), save_table(summary, "inv6"),
             save_table(table.round(4), "inv6_logit")]
    print_key_numbers(data, chi, coefs, auc, probs)
    return paths
