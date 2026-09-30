"""Investigation 5: Where are the fatal and serious crash hotspots?

Maps fatal and serious injury (FSI) crashes with hexbin density maps of
Victoria and Melbourne, ranks local government areas (LGAs) by FSI crashes,
and lists the ~1 km cells with the most FSI crashes.

Outputs:
  figures/inv5_hotspot_map.png     - hexbin maps: Victoria + Melbourne zoom
  figures/inv5_top_lgas.png        - stacked bar of the top 10 LGAs
  results/inv5.csv                 - all LGAs ranked by FSI crashes
  results/inv5_hotspot_cells.csv   - top 20 ~1 km cells by FSI crashes
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.data_loader import VIC_LAT_RANGE, VIC_LON_RANGE
from src.inv1_speed_zone import FATAL, NON_INJURY, SERIOUS, wilson_ci
from src.utils import SEVERITY_COLOURS, label_axes, save_fig, save_table

TOP_N = 10
# Moreland City Council was renamed Merri-bek in 2022; the data uses both.
LGA_RENAMES = {"MORELAND": "MERRI-BEK"}
LGA_LABELS = {"MERRI-BEK": "Merri-bek (formerly Moreland)"}
# Zoom box for the Melbourne panel: (lon_min, lon_max, lat_min, lat_max).
MELBOURNE_BOX = (144.4, 145.6, -38.35, -37.45)
# Grid for hotspot cells: 0.01 degrees is about 1.1 km north-south and
# 0.9 km east-west at Victoria's latitude.
CELL_DEG = 0.01
KM_PER_DEG_LAT = 111.0
# Blue (Okabe-Ito) contrasts with the yellow-red hexbin colour scale.
FATAL_DOT_COLOUR = "#0072B2"


def km_per_deg_lon(lat):
    return KM_PER_DEG_LAT * np.cos(np.radians(lat))


def lga_label(name):
    return LGA_LABELS.get(name, name.title())


def preprocess(df):
    """Return injury crashes with valid coordinates and merged LGA names."""
    n_start = len(df)
    print(f"  Starting rows: {n_start:,}")

    # Step 1 - Drop the incomplete period (Sep 2025 onwards).
    # Severe crashes are entered first, so these months over-represent FSI
    # crashes (same as earlier investigations).
    incomplete = df["INCOMPLETE_PERIOD"]
    print(f"  Step 1: removed {int(incomplete.sum()):,} crashes from the "
          "incomplete period")
    df = df[~incomplete]

    # Step 2 - Remove the 'Non injury accident' records (dataset only covers
    # injury crashes).
    non_injury = df["SEVERITY"] == NON_INJURY
    print(f"  Step 2: removed {int(non_injury.sum()):,} non-injury crashes")
    df = df[~non_injury]

    # Step 3 - Remove invalid coordinates.
    # A crash with no latitude/longitude can't be mapped. Zero values are a
    # common 'missing' placeholder, and anything outside Victoria's bounding
    # box must be a recording error. (Checked before writing this: only the
    # missing case occurs; exact duplicate locations are kept because many
    # crashes happen at the same intersection.)
    lat, lon = df["LATITUDE"], df["LONGITUDE"]
    missing = lat.isna() | lon.isna()
    zero = ~missing & ((lat == 0) | (lon == 0))
    outside = ~missing & ~zero & ~(lat.between(*VIC_LAT_RANGE) & lon.between(*VIC_LON_RANGE))
    invalid = missing | zero | outside
    fsi = df["SEVERITY"].isin([FATAL, SERIOUS])
    print(f"  Step 3: removed {int(invalid.sum()):,} crashes with invalid coordinates "
          f"({int((invalid & fsi).sum()):,} of them fatal/serious):")
    print(f"            missing lat/lon           {int(missing.sum()):>6,}")
    print(f"            zero lat/lon              {int(zero.sum()):>6,}")
    print(f"            outside Victoria box      {int(outside.sum()):>6,}  "
          f"(lat {VIC_LAT_RANGE}, lon {VIC_LON_RANGE})")
    df = df[~invalid].copy()

    # Step 4 - Merge renamed LGAs.
    # Moreland became Merri-bek in 2022 and records use both names, which
    # would split one council across two rows of the ranking.
    renamed = df["LGA_NAME"].isin(LGA_RENAMES)
    df["LGA_NAME"] = df["LGA_NAME"].replace(LGA_RENAMES)
    print(f"  Step 4: relabelled {int(renamed.sum()):,} crashes from "
          + ", ".join(f"{a} to {b}" for a, b in LGA_RENAMES.items())
          + " (council renamed in 2022)")

    n_fsi = int(df["SEVERITY"].isin([FATAL, SERIOUS]).sum())
    print(f"  Rows used: {len(df):,} injury crashes ({n_fsi:,} fatal or serious, "
          "used for the maps)")
    return df


def summarise_lgas(df):
    """All LGAs ranked by number of fatal + serious crashes."""
    no_lga = df["LGA_NAME"].isna()
    print(f"  LGA ranking: {int(no_lga.sum()):,} crashes with no LGA name left out "
          f"({int((no_lga & df['SEVERITY'].isin([FATAL, SERIOUS])).sum())} fatal/serious; "
          "still shown on the maps)")
    d = df[~no_lga]

    g = d.groupby("LGA_NAME")
    table = pd.DataFrame({
        "crashes": g.size(),
        "fatal": g["SEVERITY"].apply(lambda s: (s == FATAL).sum()),
        "serious": g["SEVERITY"].apply(lambda s: (s == SERIOUS).sum()),
    })
    table["fsi"] = table["fatal"] + table["serious"]
    table["pct_fsi"] = table["fsi"] / table["crashes"] * 100
    table["fsi_ci_low"], table["fsi_ci_high"] = wilson_ci(table["fsi"], table["crashes"])
    table["pct_of_state_fsi"] = table["fsi"] / table["fsi"].sum() * 100
    table["pct_of_state_fatal"] = table["fatal"] / table["fatal"].sum() * 100
    table = table.sort_values(["fsi", "fatal"], ascending=False)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table["fatal_rank"] = table["fatal"].rank(ascending=False, method="min").astype(int)
    return table.rename_axis("lga").reset_index().round(2)


def hotspot_cells(fsi, top=20):
    """Group FSI crashes into ~1 km cells and return the busiest cells."""
    d = fsi.copy()
    d["cell_lat"] = (d["LATITUDE"] / CELL_DEG).round().astype(int)
    d["cell_lon"] = (d["LONGITUDE"] / CELL_DEG).round().astype(int)
    d["road"] = (d["ROAD_NAME"].fillna("") + " " + d["ROAD_TYPE"].fillna("")).str.strip()

    def top_roads(s):
        roads = s[s != ""].value_counts().head(2)
        return "; ".join(f"{r} ({n})" for r, n in roads.items())

    g = d.groupby(["cell_lat", "cell_lon"])
    cells = pd.DataFrame({
        "fsi": g.size(),
        "fatal": g["SEVERITY"].apply(lambda s: (s == FATAL).sum()),
        "lga": g["LGA_NAME"].agg(lambda s: s.mode().iat[0] if s.notna().any() else ""),
        "main_roads": g["road"].agg(top_roads),
    }).sort_values(["fsi", "fatal"], ascending=False).head(top).reset_index()
    cells["centre_lat"] = (cells["cell_lat"] * CELL_DEG).round(3)
    cells["centre_lon"] = (cells["cell_lon"] * CELL_DEG).round(3)
    cells.insert(0, "rank", np.arange(1, len(cells) + 1))
    return cells[["rank", "centre_lat", "centre_lon", "fsi", "fatal", "lga", "main_roads"]]


def _hexbin_panel(ax, fsi, extent, gridsize, title):
    """Hexbin of FSI crashes within extent, with fatal crashes as dots."""
    lon_min, lon_max, lat_min, lat_max = extent
    hb = ax.hexbin(fsi["LONGITUDE"], fsi["LATITUDE"], gridsize=gridsize,
                   extent=extent, bins="log", mincnt=1, cmap="YlOrRd",
                   linewidths=0.1, edgecolors="face")
    # Fatal crashes in blue with a thin white edge so they stand out from the
    # yellow-red hexagons, including the dark red city cells.
    fatal = fsi[fsi["SEVERITY"] == FATAL]
    ax.scatter(fatal["LONGITUDE"], fatal["LATITUDE"], s=3, color=FATAL_DOT_COLOUR,
               edgecolors="white", linewidths=0.15, alpha=0.8, rasterized=True)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    # Correct for longitude degrees being shorter than latitude degrees.
    mid_lat = (lat_min + lat_max) / 2
    ax.set_aspect(1 / np.cos(np.radians(mid_lat)))
    ax.grid(alpha=0.2)
    label_axes(ax, title, "Longitude (degrees east)",
               "Latitude (degrees, negative = south)")
    hex_km = (lon_max - lon_min) / gridsize * km_per_deg_lon(mid_lat)
    return hb, hex_km


def plot_map(fsi):
    fig, (vic, mel) = plt.subplots(1, 2, figsize=(18, 7.5),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    vic_extent = (VIC_LON_RANGE[0], VIC_LON_RANGE[1], -39.25, -33.95)
    hb1, km1 = _hexbin_panel(vic, fsi, vic_extent, 110,
                             "Fatal and serious injury crashes, Victoria")
    hb2, km2 = _hexbin_panel(mel, fsi, MELBOURNE_BOX, 55,
                             "Melbourne and surrounds")

    lon_min, lon_max, lat_min, lat_max = MELBOURNE_BOX
    box_style = dict(fill=False, edgecolor="black", linewidth=1.2, linestyle="--")
    vic.add_patch(Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                            **box_style))

    # Explicit ticks so the last Melbourne label (145.6) isn't cut off.
    mel.set_xticks(np.arange(lon_min, lon_max + 0.01, 0.2))
    mel.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

    for ax, hb in ((vic, hb1), (mel, hb2)):
        # Extra padding so the colour bar doesn't cover the last x tick label.
        cbar = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.05)
        cbar.set_label("Fatal + serious crashes per hexagon\n(count, log scale)")
    legend_items = [
        Line2D([], [], marker="o", color=FATAL_DOT_COLOUR, markeredgecolor="white",
               linestyle="", markersize=5, label="Fatal crash"),
        Rectangle((0, 0), 1, 1, label="Melbourne panel area", **box_style),
    ]
    legend_style = dict(frameon=True, framealpha=0.85, facecolor="white",
                        edgecolor="none")
    vic.legend(handles=legend_items, loc="upper right", **legend_style)
    # Anchor the Melbourne legend over Port Phillip Bay, which has no crashes.
    mel.legend(handles=legend_items[:1], loc="upper left",
               bbox_to_anchor=(144.72, -38.02), bbox_transform=mel.transData,
               **legend_style)

    fig.text(0.01, -0.02,
             f"Injury crashes, Jan 2012 – Aug 2025 ({len(fsi):,} fatal or serious). "
             f"Hexagon width ≈ {km1:.0f} km (Victoria) and ≈ {km2:.1f} km (Melbourne). "
             "No base map: crash locations trace the road network.",
             fontsize=9, ha="left")
    return save_fig(fig, "inv5_hotspot_map")


def plot_top_lgas(table):
    top = table.head(TOP_N).iloc[::-1]  # reversed so rank 1 is at the top
    labels = [lga_label(l) for l in top["lga"]]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(labels, top["fatal"], color=SEVERITY_COLOURS[FATAL], label="Fatal")
    ax.barh(labels, top["serious"], left=top["fatal"],
            color=SEVERITY_COLOURS[SERIOUS], label="Serious injury")
    for y, r in enumerate(top.itertuples()):
        ax.text(r.fsi + 40, y, f"{r.fsi:,}  ({r.pct_of_state_fsi:.1f}% of state; "
                f"{r.fatal} fatal)", va="center", fontsize=9)
    ax.set_xlim(0, top["fsi"].max() * 1.45)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", visible=False)
    label_axes(ax, f"Top {TOP_N} local government areas by fatal and serious crashes",
               "Crashes (count, Jan 2012 – Aug 2025)", "Local government area")
    ax.legend(loc="lower right")
    fig.text(0.01, -0.02, "Counts, not rates: busier and more populous LGAs "
             "naturally have more crashes.", fontsize=9, ha="left")
    return save_fig(fig, "inv5_top_lgas")


def print_key_numbers(df, fsi, table, cells):
    print("\n  Key numbers:")
    print(f"  - {len(fsi):,} fatal or serious crashes mapped, Jan 2012 – Aug 2025 "
          f"({int((fsi['SEVERITY'] == FATAL).sum()):,} fatal)")

    print(f"\n  Top {TOP_N} LGAs by fatal + serious crashes:")
    for r in table.head(TOP_N).itertuples():
        print(f"    {r.rank:>2}. {lga_label(r.lga):<28} {r.fsi:>5,} FSI "
              f"({r.pct_of_state_fsi:4.1f}% of state), {r.fatal:>3} fatal, "
              f"{r.pct_fsi:.1f}% of its crashes FSI")
    share = table.head(TOP_N)["pct_of_state_fsi"].sum()
    print(f"    -> top {TOP_N} of {len(table)} LGAs account for {share:.1f}% of "
          "the state's fatal + serious crashes")

    print(f"\n  Top {TOP_N} LGAs by fatal crashes:")
    by_fatal = table.sort_values(["fatal", "fsi"], ascending=False).head(TOP_N)
    for i, r in enumerate(by_fatal.itertuples(), 1):
        note = "" if r.rank <= TOP_N else f"  <- only #{r.rank} by fatal + serious"
        print(f"    {i:>2}. {lga_label(r.lga):<28} {r.fatal:>3} fatal "
              f"({r.pct_of_state_fatal:.1f}% of state){note}")

    for label, mask in [("Fatal + serious", fsi["SEVERITY"].isin([FATAL, SERIOUS])),
                        ("Fatal", fsi["SEVERITY"] == FATAL)]:
        div = fsi.loc[mask, "STAT_DIV_NAME"].value_counts(normalize=True) * 100
        print(f"  - {label} crashes: {div.get('Metro', 0):.1f}% metro Melbourne, "
              f"{div.get('Country', 0):.1f}% country Victoria")
    all_div = df["STAT_DIV_NAME"].value_counts(normalize=True) * 100
    print(f"    (all injury crashes: {all_div.get('Metro', 0):.1f}% metro, "
          f"{all_div.get('Country', 0):.1f}% country)")

    print("\n  Top 5 hotspot cells (~1 km):")
    for r in cells.head(5).itertuples():
        print(f"    {r.rank}. ({r.centre_lat}, {r.centre_lon}) {lga_label(r.lga)}: "
              f"{r.fsi} FSI, {r.fatal} fatal - {r.main_roads}")
    print("  - Caveat: counts, not rates. There is no population or traffic data, so "
          "the ranking shows where severe crashes happen, not where travel is "
          "most dangerous per km driven.")


def run(df):
    """Run investigation 5. Returns the list of saved file paths."""
    data = preprocess(df)
    fsi = data[data["SEVERITY"].isin([FATAL, SERIOUS])]
    table = summarise_lgas(data)
    cells = hotspot_cells(fsi)
    paths = [plot_map(fsi), plot_top_lgas(table),
             save_table(table, "inv5"), save_table(cells, "inv5_hotspot_cells")]
    print_key_numbers(data, fsi, table, cells)
    return paths
