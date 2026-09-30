# Victorian Road Crash Data – Road Safety Analysis

Python analysis of Victorian road crash data (Jan 2012 – Jan 2026, 200,754 crash
records). It covers data quality, speed zones, time of day, yearly trends, light
conditions, crash hotspots, statistical models, and a deeper look at whether the
speed-zone effect holds up once other factors are controlled for.

## Requirements

- **Python 3.9 or newer** (tested with 3.9 on macOS; the lab computers use 3.13)
- The packages in `requirements.txt`: numpy, pandas, matplotlib, scikit-learn, scipy
- **The data file** at `data/victorian_road_crash_data.csv`. It is not included
  in git (`/data` is in `.gitignore`), so copy it into a `data/` folder in the
  project root before running.

## Setup

### macOS / Linux (virtual environment)

```bash
cd Assessment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows lab computers (conda)

Open **Anaconda Prompt**, then:

```bat
conda create --name myenv python=3.13
conda activate myenv
cd path\to\Assessment
pip install -r requirements.txt
```

If `myenv` already exists (`conda info --envs`), skip the `create` step.

Do not copy the `.venv` folder between computers – it only works on the machine
it was created on. Recreate the environment on each machine instead.

## Running

With the environment active, from the project folder:

```bash
python main.py                 # run all investigations (~20 seconds)
python main.py > output.txt    # same, saving the printed results to a file
python explore.py              # initial exploration: shape, dtypes, missing values, value counts
```

`main.py` loads and cleans the data once, then runs every investigation in
order. For each one it prints the preprocessing steps (and how many rows each
removed), the key numbers for the report, and the files it saved. It creates
`figures/` (PNG, 300 dpi) and `results/` (CSV) if they don't exist and
overwrites their contents, so they can be deleted and rebuilt at any time.

## Project structure

```
Assessment/
├── main.py                      # runs all investigations in order
├── explore.py                   # initial data exploration (prints only)
├── requirements.txt
├── main_output.txt              # printout of `python main.py` (source of the report's numbers)
├── explore_output.txt           # printout of `python explore.py`
├── report_notes.md              # per-investigation notes: question, preprocessing, numbers, interpretation, limitations
├── report.docx                  # final engineering report (figures embedded from figures/)
├── data/                        # input CSV (not in git)
├── figures/                     # generated PNGs
├── results/                     # generated CSV tables
└── src/
    ├── data_loader.py           # load, merge (on ACCIDENT_NO) and clean the data
    ├── utils.py                 # save_fig, save_table, shared plot style
    ├── investigations/
    │   └── data_quality.py      # investigation 0
    ├── inv1_speed_zone.py       # investigation 1
    ├── inv2_time_of_day.py      # investigation 2
    ├── inv3_yearly_trend.py     # investigation 3
    ├── inv4_light_condition.py  # investigation 4
    ├── inv5_hotspots.py         # investigation 5
    ├── inv6_severity_model.py   # investigation 6
    └── deep_investigation.py    # investigation 7
```

## Investigations and outputs

All figures are saved to `figures/` and all tables to `results/`.

| # | Script | Question | Figures (`figures/`) | Tables (`results/`) |
|---|---|---|---|---|
| 0 | `src/investigations/data_quality.py` | What does the data look like, and what are its quality issues? | `dq_01_crashes_per_month.png`, `dq_02_missing_values.png`, `dq_03_unknown_labels.png`, `dq_04_crash_locations.png`, `dq_05_categories.png` | – |
| 1 | `src/inv1_speed_zone.py` | How does speed zone influence crash severity? | `inv1_speed_zone_severity.png` | `inv1.csv` |
| 2 | `src/inv2_time_of_day.py` | How does crash frequency vary by hour of day and day of week? | `inv2_heatmap_hour_day.png`, `inv2_hourly_profile.png` | `inv2.csv` |
| 3 | `src/inv3_yearly_trend.py` | What is the yearly crash trend (2012–2024)? | `inv3_yearly_trend.png` | `inv3.csv` |
| 4 | `src/inv4_light_condition.py` | How does light condition relate to the fatality rate? | `inv4_light_condition.png` | `inv4.csv`, `inv4_by_speed.csv` |
| 5 | `src/inv5_hotspots.py` | Where are the fatal and serious crash hotspots? | `inv5_hotspot_map.png`, `inv5_top_lgas.png` | `inv5.csv`, `inv5_hotspot_cells.csv` |
| 6 | `src/inv6_severity_model.py` | Chi-square test and logistic regression: what predicts a fatal or serious crash? | `inv6_severity_model.png` | `inv6.csv`, `inv6_logit.csv` |
| 7 | `src/deep_investigation.py` | Does the speed-zone effect hold after controlling for other factors? | `deep_01_stratified_gradient.png`, `deep_02_model_attenuation.png`, `deep_03_stratum_or.png` | `deep_stratified.csv`, `deep_models.csv` |

## Data processing

Cleaning is done once in `src/data_loader.py`; each investigation then prints
the extra filtering it applies.

- **Incomplete recent data:** crashes from Sep 2025 onwards are flagged
  (`INCOMPLETE_PERIOD`) and excluded from analyses. Those months are under-counted
  and skewed towards severe crashes, which are entered first.
- **Unknown codes → missing:** labels such as "Not known", "Unk." and "Unknown",
  and route codes 9999 / -1, are treated as missing values.
- **Speed zone** is converted to a number (`SPEED_ZONE_KMH`); "Camping grounds or
  off road" and "Other speed limit" have no km/h value.
- **Non-injury crashes** (4 records) are removed – the dataset only covers
  injury crashes.
- **LGA rename:** Moreland (renamed Merri-bek in 2022) is merged into Merri-bek
  in the hotspot analysis.
- **Joining tables:** the current extract is a single crash-level file. If
  separate person or vehicle files are added to `data/`, `data_loader.py`
  aggregates them to one row per `ACCIDENT_NO` before merging, so crashes are
  not duplicated.

## Adding an investigation

1. Create a module in `src/` with a `run(df)` function that saves its figures
   with `save_fig(fig, name)` and tables with `save_table(table, name)`, and
   returns the list of saved paths.
2. Add it to the `INVESTIGATIONS` list in `main.py`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `zsh: command not found: python` (macOS) | Activate the environment (`source .venv/bin/activate`), or use `python3` / `.venv/bin/python`. |
| `ModuleNotFoundError: No module named 'pandas'` | The environment isn't active, or packages aren't installed: activate it and run `pip install -r requirements.txt`. |
| `FileNotFoundError: No crash-level CSV found` | Put `victorian_road_crash_data.csv` in the `data/` folder. |
