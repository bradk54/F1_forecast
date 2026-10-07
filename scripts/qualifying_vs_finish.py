"""Where a driver qualifies versus where they finish, over every season we hold.

Writes two figures and one table to ``Reports/figures/``:

* ``qualifying_vs_finish_heatmap.png``  P(finish position | qualifying position)
* ``qualifying_vs_finish_slots.png``    the same distribution for selected slots
* ``qualifying_vs_finish.csv``          the per-slot numbers behind both

    F1_DATA_DIR=<main>/Data python -m scripts.qualifying_vs_finish

Choices that change the answer, and why they were made:

* **x is the qualifying classification, not the starting grid.** The grid is what
  the race starts from, but it is qualifying after penalties, pit-lane starts and
  (2021-22) sprint results, and one penalty re-numbers every car behind it. The
  two disagree on roughly a third of rows. The race tables (``finishes.csv``) and
  qualifying tables (``qualifying.csv``) cover 2000-2024; ``race_results.parquet``
  carries 2025-26 but only as a grid, so those seasons enter the cross-check
  (``grid_crosscheck``) and not the headline.
* **A retirement is its own outcome, not a finishing position.** Averaging a DNF
  in as "last" or dropping it both misstate the distribution, so each row of the
  heatmap is the full outcome distribution: P1..P20 plus ``Out``. The *average
  finish* is over classified finishers only and is always reported next to the
  retirement rate, because the two trade off (a back-marker who keeps retiring
  has a flattering average).
* **DSQ / DNS / DNQ / DNP / EX are not outcomes of the slot.** A disqualified car
  finished on track and a non-starter never raced, so neither is attrition.
* **Qualifying slots above 20 are dropped** (4.5% of starters, from the 22-26 car
  eras): the slots are too thin to read and would stretch the axis for everyone.
  Another 1.3% have no qualifying time at all (107% rule, exclusion), so no slot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FIGURES_DIR, PROCESSED_DIR, RACE_RESULTS_PATH  # noqa: E402

FIRST_SEASON, LAST_CSV_SEASON = 2000, 2024
MAX_SLOT = 20  # qualifying slots and finishing columns shown
SLOTS_SHOWN = (1, 2, 3, 5, 10, 15)

NOT_AN_OUTCOME = {"DSQ", "DQ", "DNS", "DNQ", "DNP", "EX"}  # race table codes

# Chart ink and surface (light), and the single-hue blue ramp, from the dataviz
# reference palette. The ramp starts at the surface so a zero cell recedes.
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE = "#2a78d6"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]


# --------------------------------------------------------------------------- load


def _first_race_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first race under each (year, slug), dropping a mislabelled second.

    ``2024/finishes.csv`` and ``2024/qualifying.csv`` each hold two events under
    the ``australian`` and ``chinese`` slugs: the real 2024 race and the 2025 race,
    written with ``race_year=2024``. Nothing errors -- the rows are valid -- but a
    join on (year, race, driver) pairs 2025 results with 2024 qualifying for every
    driver who raced in both. A second pole-sitter inside one event is the tell,
    since each real event has exactly one P1.
    """
    starts = df["Pos."].astype(str).eq("1")
    nth_race = starts.groupby([df["race_year"], df["race"]]).cumsum()
    return df[nth_race <= 1]


def _load_csv_track(name: str) -> pd.DataFrame:
    frames = [
        pd.read_csv(PROCESSED_DIR / str(y) / f"{name}.csv")
        for y in range(FIRST_SEASON, LAST_CSV_SEASON + 1)
    ]
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = _first_race_only(df).copy()
    if before - len(df):
        print(f"  {name}.csv: dropped {before - len(df)} rows of a mislabelled second race")
    df["driver"] = df["Driver"].str.replace(r"^#\d+", "", regex=True).str.strip()

    winners = df["Pos."].astype(str).eq("1").groupby([df["race_year"], df["race"]]).sum()
    assert winners.max() == 1, f"{name}: an event still has {winners.max()} P1 rows"
    return df


def load_races() -> pd.DataFrame:
    """One row per driver per race, 2000-2024: constructor, quali, grid, finish, outcome."""
    fin, qual = _load_csv_track("finishes"), _load_csv_track("qualifying")

    # one_to_one makes pandas raise on a repeated key rather than multiply rows.
    df = fin.merge(
        qual[["race_year", "race", "driver", "Pos."]].rename(columns={"Pos.": "quali_raw"}),
        on=["race_year", "race", "driver"],
        how="left",
        validate="one_to_one",
    )
    df["year"] = df["race_year"]
    df["constructor"] = df["Constructor"]
    df["quali"] = pd.to_numeric(df["quali_raw"], errors="coerce")
    df["grid"] = pd.to_numeric(df["Grid"].astype(str).str.extract(r"(\d+)")[0])  # 'PL' -> NaN
    df["finish"] = pd.to_numeric(df["Pos."], errors="coerce")
    df["outcome"] = np.select(
        [df["finish"].notna(), df["Pos."].isin(NOT_AN_OUTCOME)],
        ["finished", "excluded"],
        default="out",  # DNF and NC
    )
    return df[["year", "race", "driver", "constructor", "quali", "grid", "finish", "outcome"]]


def load_recent_grid() -> pd.DataFrame:
    """2025-26 races from the parquet, as (grid, finish, outcome) -- no quali."""
    rr = pd.read_parquet(RACE_RESULTS_PATH)
    rr = rr[(rr["session_type"] == "R") & (rr["Year"] > LAST_CSV_SEASON)]
    code = rr["ClassifiedPosition"].astype(str)
    finish = pd.to_numeric(code, errors="coerce")
    return pd.DataFrame(
        {
            "year": rr["Year"].to_numpy(),
            "grid": rr["GridPosition"].where(rr["GridPosition"] > 0).to_numpy(),  # 0 = pit lane
            "finish": finish.to_numpy(),
            "outcome": np.select(
                [finish.notna(), code.eq("R")], ["finished", "out"], default="excluded"
            ),
        }
    )


# ------------------------------------------------------------------------ analyse


def outcome_matrix(df: pd.DataFrame, slot_col: str) -> pd.DataFrame:
    """Rows: slot 1..20. Columns: finish 1..20 (20 means 20+), then ``Out``.

    Row-normalised, so each row is P(outcome | slot) and sums to 1. Positions past
    20 fold into the last column for display only; ``slot_summary`` uses the true
    finishing positions.
    """
    d = df[df["outcome"].isin(["finished", "out"]) & df[slot_col].between(1, MAX_SLOT)]
    col = np.where(d["outcome"] == "out", MAX_SLOT + 1, d["finish"].clip(upper=MAX_SLOT))
    counts = pd.crosstab(d[slot_col].astype(int), pd.Series(col, index=d.index).astype(int))
    counts = counts.reindex(index=range(1, MAX_SLOT + 1), columns=range(1, MAX_SLOT + 2), fill_value=0)
    counts.columns = [*range(1, MAX_SLOT), f"{MAX_SLOT}+", "Out"]
    return counts.div(counts.sum(axis=1), axis=0)


def slot_summary(df: pd.DataFrame, slot_col: str) -> pd.DataFrame:
    d = df[df["outcome"].isin(["finished", "out"]) & df[slot_col].between(1, MAX_SLOT)]
    g = d.groupby(slot_col)
    fin = d[d["outcome"] == "finished"].groupby(slot_col)["finish"]
    return pd.DataFrame(
        {
            "n": g.size(),
            "avg_finish": fin.mean(),
            "median_finish": fin.median(),
            "p25_finish": fin.quantile(0.25),
            "p75_finish": fin.quantile(0.75),
            "win_pct": g["finish"].apply(lambda s: (s == 1).mean() * 100),
            "podium_pct": g["finish"].apply(lambda s: (s <= 3).mean() * 100),
            "points_pct": g["finish"].apply(lambda s: (s <= 10).mean() * 100),
            "out_pct": g["outcome"].apply(lambda s: (s == "out").mean() * 100),
        }
    ).rename_axis("quali_slot")


def grid_crosscheck(races: pd.DataFrame, recent: pd.DataFrame, headline: pd.DataFrame) -> None:
    """Does the answer survive swapping qualifying for the grid and adding 2025-26?"""
    grid_all = pd.concat([races[["year", "grid", "finish", "outcome"]], recent], ignore_index=True)
    by_grid = slot_summary(grid_all, "grid")
    diff = (by_grid["avg_finish"] - headline["avg_finish"]).abs()
    both = races["quali"].notna() & races["grid"].notna()
    print("\nCross-check: start grid, 2000-2026, instead of qualifying, 2000-2024")
    print(f"  quali != grid on {(races.loc[both, 'quali'] != races.loc[both, 'grid']).mean():.0%} of rows")
    print(f"  rows {int(by_grid['n'].sum()):,} (+{len(recent):,} from 2025-26)")
    print(f"  average finish differs by at most {diff.max():.2f} places (slot {int(diff.idxmax())}), "
          f"median {diff.median():.2f}")
    print(f"  pole: {by_grid.loc[1, 'win_pct']:.1f}% win from grid vs {headline.loc[1, 'win_pct']:.1f}% from quali")


# --------------------------------------------------------------------------- plot


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def heat_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("blue_seq", [SURFACE, *RAMP])


def draw_heatmap(ax, prob: pd.DataFrame, summary: pd.DataFrame, vmax: float, *,
                 compact: bool = False, marker_size: float = 34):
    """Draw the slot-by-outcome grid on ``ax`` and return the mappable.

    Columns 1..20 sit at x = 1..20; "Out" gets its own cell at x = 22, past a gap,
    because it is an outcome and not a finishing position. ``compact`` thins the
    ticks for small multiples. The mean line is drawn in the same units as the
    cells (x = finishing position, y = qualifying slot), so it is not a second axis.
    """
    vals = prob.to_numpy() * 100
    out_x = MAX_SLOT + 2
    y_edges = np.arange(0.5, MAX_SLOT + 1.5)
    kw = dict(cmap=heat_cmap(), vmin=0, vmax=vmax, edgecolors=SURFACE, linewidth=1.0)
    mappable = ax.pcolormesh(np.arange(0.5, MAX_SLOT + 1.5), y_edges, vals[:, :MAX_SLOT], **kw)
    ax.pcolormesh([out_x - 0.5, out_x + 0.5], y_edges, vals[:, MAX_SLOT:], **kw)

    ring = [patheffects.withStroke(linewidth=2.2, foreground=SURFACE)]
    ax.plot([0.5, MAX_SLOT + 0.5], [0.5, MAX_SLOT + 0.5], color=INK_2, lw=1, ls=(0, (4, 3)), zorder=3)
    ax.plot(summary["avg_finish"], summary.index, color=INK, lw=1.6, zorder=4, path_effects=ring)
    ax.scatter(summary["avg_finish"], summary.index, s=marker_size, color=INK, zorder=5,
               edgecolor=SURFACE, linewidth=1.5)

    ax.set_xlim(0.5, out_x + 0.5)
    ax.set_ylim(MAX_SLOT + 0.5, 0.5)
    shown = [1, 5, 10, 15, MAX_SLOT] if compact else list(range(1, MAX_SLOT + 1))
    ax.set_xticks([*shown, out_x])
    ax.set_xticklabels([*[f"{MAX_SLOT}+" if i == MAX_SLOT else str(i) for i in shown], "Out"], fontsize=9)
    ax.set_yticks(shown if compact else range(1, MAX_SLOT + 1))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return mappable


def plot_heatmap(prob: pd.DataFrame, summary: pd.DataFrame, n_rows: int, path: Path) -> None:
    vals = prob.to_numpy() * 100
    vmax = float(np.ceil(vals.max() / 5) * 5)

    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    main = draw_heatmap(ax, prob, summary, vmax)
    ax.set_xlabel("Finishing position in the race")
    ax.set_ylabel("Qualifying position")

    pole, p2 = summary.loc[1], summary.loc[2]
    fig.text(0.07, 0.955, f"Pole wins {pole['win_pct']:.0f}% of races; a P2 qualifier averages P{p2['avg_finish']:.1f}",
             fontsize=15, fontweight="bold", ha="left")
    fig.text(0.07, 0.918, f"Share of cars from each qualifying slot reaching each outcome · 2000–2024 · "
             f"{n_rows:,} car-starts · each row sums to 100%", fontsize=10.5, color=INK_2, ha="left")

    cb = fig.colorbar(main, ax=ax, fraction=0.035, pad=0.02, extend="max" if vals.max() > vmax else "neither")
    cb.set_label("% of cars from that qualifying slot", color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9)
    cb.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")

    handles = [
        plt.Line2D([0], [0], color=INK, lw=1.6, marker="o", ms=5, label="Average finish (finishers only)"),
        plt.Line2D([0], [0], color=INK_2, lw=1, ls=(0, (4, 3)), label="Finish = qualify"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.07, 0.005), ncol=2, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.87, bottom=0.12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_slots(prob: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharey=True)
    ymax = float(np.ceil(prob.loc[list(SLOTS_SHOWN)].to_numpy().max() * 100 / 10) * 10)
    x = np.arange(1, MAX_SLOT + 1)
    for ax, slot in zip(axes.flat, SLOTS_SHOWN):
        row = prob.loc[slot].to_numpy() * 100
        ax.bar(x, row[:MAX_SLOT], width=0.72, color=BLUE, zorder=2)
        ax.bar(MAX_SLOT + 2.2, row[MAX_SLOT], width=0.72, color=MUTED, zorder=2)
        ax.axvline(slot, color=MUTED, lw=1, zorder=1)
        ax.axvline(summary.loc[slot, "avg_finish"], color=INK, lw=1.4, ls=(0, (4, 3)), zorder=3)
        s = summary.loc[slot]
        ax.set_title(f"Qualify P{slot}", loc="left", fontsize=11, fontweight="bold", pad=17)
        ax.annotate(f"avg finish P{s['avg_finish']:.1f} · out {s['out_pct']:.0f}%", xy=(0, 1),
                    xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                    fontsize=9, color=INK_2, va="bottom", ha="left")
        ax.set_xticks([1, 5, 10, 15, 20, MAX_SLOT + 2.2])
        ax.set_xticklabels(["1", "5", "10", "15", "20", "Out"], fontsize=8.5)
        ax.set_ylim(0, ymax)
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    fig.supxlabel("Finishing position in the race", color=INK_2, fontsize=10)
    fig.text(0.04, 0.965, "Where a car finishes, given where it qualified", fontsize=15, fontweight="bold", ha="left")
    fig.text(0.04, 0.93, "Share of cars at each finishing position · 2000–2024", fontsize=10.5, color=INK_2, ha="left")
    handles = [
        plt.Line2D([0], [0], color=MUTED, lw=1, label="Qualifying slot"),
        plt.Line2D([0], [0], color=INK, lw=1.4, ls=(0, (4, 3)), label="Average finish (finishers only)"),
    ]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.98, 0.0), ncol=2, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.14, wspace=0.08, hspace=0.45)
    fig.savefig(path, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def main() -> int:
    print(f"Reading {PROCESSED_DIR}")
    races = load_races()

    n_total = len(races)
    started = races["outcome"].isin(["finished", "out"])
    has_slot = races["quali"].between(1, MAX_SLOT)
    used = races[started & has_slot]
    print(f"\n{n_total:,} car-race rows, {races.groupby(['year', 'race']).ngroups} races, "
          f"{races['year'].min()}-{races['year'].max()}")
    print(f"  not an outcome of the slot (DSQ/DNS/DNQ/DNP/EX): {(~started).sum():,}")
    print(f"  started, qualified beyond P{MAX_SLOT}: {(started & (races['quali'] > MAX_SLOT)).sum():,}")
    print(f"  started, no qualifying slot (no time or excluded): {(started & races['quali'].isna()).sum():,}")
    print(f"  used: {len(used):,}")

    prob = outcome_matrix(races, "quali")
    summary = slot_summary(races, "quali")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    plot_heatmap(prob, summary, len(used), FIGURES_DIR / "qualifying_vs_finish_heatmap.png")
    plot_slots(prob, summary, FIGURES_DIR / "qualifying_vs_finish_slots.png")
    summary.round(2).to_csv(FIGURES_DIR / "qualifying_vs_finish.csv")

    print("\nPer qualifying slot (average finish is over classified finishers only):")
    print(summary.round(1).to_string())

    era = used.assign(era=pd.cut(used["year"], [1999, 2009, 2019, 2024], labels=["2000-09", "2010-19", "2020-24"]))
    print("\nRetirement rate by era (why pooling all seasons flatters nobody evenly):")
    print((era.groupby("era", observed=True)["outcome"].apply(lambda s: (s == "out").mean() * 100)).round(1).to_string())

    grid_crosscheck(races, load_recent_grid(), summary)
    print(f"\nWrote figures and table to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
