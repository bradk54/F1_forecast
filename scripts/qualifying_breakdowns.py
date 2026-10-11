"""Qualifying versus finish, broken out by era, driver, constructor, pairing and Grand Prix.

Builds on ``scripts.qualifying_vs_finish`` -- same 2000-2024 data, same exclusions
(DSQ/DNS/DNQ/DNP/EX out; qualifying slots above 20 or with no time out).

    F1_DATA_DIR=<main>/Data python -m scripts.qualifying_breakdowns

Writes ``qualifying_by_*.png`` and ``qualifying_by_*.csv`` to ``Reports/figures/``.

How groups are compared, and why:

* **Against an expectation, never a raw average.** A driver who usually qualifies
  15th finishes worse than one who qualifies 3rd whatever their skill, and a 2003
  car retired three times as often as a 2023 one. So each start is scored against
  what a *comparable car starting there in that era* did: ``gain`` is expected
  finish minus actual finish (finishers only; positive = beat the expectation), and
  ``out_avoided_pp`` is expected minus actual retirement, in percentage points. A
  group's number is the mean of its starts.
* **"Comparable" means the same slot *and* the same strength of car.** Slot alone
  is a noisy proxy for pace: a typical P14 starter mixes slow cars that belong
  there with fast ones that had a bad Saturday and will recover, so against a
  slot-only baseline every habitually slow car looks like a poor racer (a
  constructor's gain correlated -0.91 with where it usually qualified: the chart
  ranked cars, not execution). The expectation therefore adds a team-pace term:
  the monotone slot curve plus ``c`` x (this car's season-mean qualifying slot,
  minus the typical team's at that slot), ``c`` fitted per era. Starting P5 in a
  Ferrari and in a Haas are different situations, and now score differently.
* **The slot curve is monotone** (isotonic fit, per era). The newest era has ~65
  cars per slot, so raw per-slot means are noisy enough to move a group's score;
  "start further back, finish no better" is the one shape we are sure of, and a
  straight line mis-fits the very front by about a third of a place.
* **Circuits are measured differently.** Averaged over a circuit, ``gain`` is zero
  by construction, so they get *grid stickiness*: per race, the rank correlation
  between qualifying and finishing order among finishers, shifted so each era has
  the same mean (otherwise a circuit that missed the refuelling years looks
  stickier than it is). Pole win rate and retirements are shown beside it.
* **Constructors follow the organisation** (Benetton -> Renault -> Alpine is one
  group), the same principle as ``src/data/teams.py``, which is keyed on Ergast IDs
  from 2018 and cannot name these 2000-2017 pitwall teams. An unmapped name raises,
  because a silent rebrand would split a team's history in two.
* **Intervals treat every start as independent.** Team-mates share a car and a
  race's incidents hit several cars at once, so the true uncertainty is somewhat
  wider than the drawn 95% whiskers. Read blue as "worth a look", not "proven".
* **A Grand Prix is not quite a circuit.** The slug is the event name: European,
  German, United States and French GPs changed venue (marked with a dagger), and
  the 2020 one-offs (Eifel, Tuscan, ...) are too few to show.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qualifying_vs_finish import (  # noqa: E402
    BLUE,
    GRID,
    INK,
    INK_2,
    MAX_SLOT,
    MUTED,
    SURFACE,
    apply_style,
    draw_heatmap,
    load_races,
    outcome_matrix,
    slot_summary,
)
from src.config import FIGURES_DIR  # noqa: E402

# TODO(human): confirm or redraw the era boundaries. Each entry is (label, first
# year, last year); every season 2000-2024 must fall in exactly one. These follow
# the regulation changes and the retirement-rate steps measured on this data
# (2000-05 ~30%, 2006-13 ~17%, 2014-21 ~15%, 2022-24 ~10%), but where an era
# starts is a judgement. Every expectation below is rebuilt per era, so changing
# this tuple re-runs the whole analysis on your definition.
ERAS: tuple[tuple[str, int, int], ...] = (
    ("2000-05", 2000, 2005),
    ("2006-13", 2006, 2013),
    ("2014-21", 2014, 2021),
    ("2022-24", 2022, 2024),
)

MIN_STARTS = {"driver": 100, "constructor": 100, "pairing": 60}
MIN_CIRCUIT_RACES = 7
MIN_FINISHERS_FOR_RHO = 8
Z95 = 1.96
VENUE_CHANGED = {"european", "german", "united-states", "french"}

# Label -> the pitwall constructor names that are one organisation. The label is
# what appears on charts: the latest name, then the earlier ones worth knowing.
LINEAGES: dict[str, list[str]] = {
    "Ferrari": ["Ferrari"],
    "McLaren": ["McLaren"],
    "Williams": ["Williams"],
    "Mercedes (ex-BAR, Honda, Brawn)": ["BAR", "Honda", "Brawn", "Mercedes"],
    "Red Bull (ex-Jaguar)": ["Jaguar", "Red Bull"],
    "Alpine (ex-Benetton, Renault, Lotus)": ["Benetton", "Renault", "Lotus", "Alpine"],
    "RB (ex-Minardi, Toro Rosso, AlphaTauri)": ["Minardi", "Toro Rosso", "AlphaTauri", "RB"],
    "Aston Martin (ex-Jordan, Force India, Racing Point)": [
        "Jordan", "Midland", "Spyker", "Force India", "Racing Point", "Aston Martin",
    ],
    "Sauber (incl. BMW Sauber, Alfa Romeo)": ["Sauber", "BMW Sauber", "Alfa Romeo", "Kick Sauber"],
    "Manor (ex-Virgin, Marussia)": ["Virgin", "Marussia", "Manor"],
    "Caterham (ex-Lotus Racing)": ["Lotus Racing", "Caterham"],
    "Toyota": ["Toyota"],
    "Haas": ["Haas"],
    "Prost": ["Prost"],
    "Arrows": ["Arrows"],
    "Super Aguri": ["Super Aguri"],
    "HRT": ["HRT"],
}
LINEAGE = {name: label for label, names in LINEAGES.items() for name in names}


def short_name(label: str) -> str:
    return label.split(" (")[0]


def era_label(year: int) -> str:
    for label, lo, hi in ERAS:
        if lo <= year <= hi:
            return label
    raise ValueError(f"season {year} is in no era; check ERAS")


def ordinal_colors(n: int) -> list:
    """n shades of blue, light to dark, none lighter than the ordinal floor."""
    ramp = LinearSegmentedColormap.from_list("era", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    return [ramp(v) for v in np.linspace(0, 1, n)]


# --------------------------------------------------------------------------- data


def prepare(races: pd.DataFrame) -> pd.DataFrame:
    """Usable starts, with era, lineage and the per-(era, slot) expectation attached."""
    d = races[races["outcome"].isin(["finished", "out"]) & races["quali"].between(1, MAX_SLOT)].copy()
    d["quali"] = d["quali"].astype(int)
    d["era"] = d["year"].map(era_label)

    unmapped = sorted(set(d["constructor"]) - set(LINEAGE))
    if unmapped:
        raise KeyError(f"constructors with no lineage entry: {unmapped}")
    d["lineage"] = d["constructor"].map(LINEAGE)
    return attach_expectation(d)


def _monotone_plus_team(e: pd.DataFrame, y: pd.Series, rows: pd.Series, dev: pd.Series, **iso_kw):
    """Slot-only monotone expectation of ``y``, then a team-pace correction.

    ``dev`` is each car's team pace minus the typical team's at that slot, so it is
    the part of the car's strength that its slot does not already reveal. ``c``
    regresses the slot-only residual on it through the origin (the residual is
    already mean-zero at every slot), fitted on ``rows`` only.
    """
    iso = IsotonicRegression(increasing=True, **iso_kw).fit(e.loc[rows, "quali"], y[rows])
    base = pd.Series(iso.predict(e["quali"]), index=e.index)
    resid, x = y[rows] - base[rows], dev[rows]
    c = float((resid * x).sum() / (x**2).sum())
    return base + c * dev, c


def attach_expectation(d: pd.DataFrame) -> pd.DataFrame:
    """Add ``exp_finish`` and ``exp_out``: what a comparable car starting there did, per era.

    ``team_pace`` is the organisation's mean qualifying slot that season, over all
    its cars (low = fast). It includes the car being scored, but a season is ~40
    starts, so that self-weight is small.
    """
    d = d.copy()
    d["team_pace"] = d.groupby(["lineage", "year"])["quali"].transform("mean")
    d["exp_finish"] = np.nan
    d["exp_out"] = np.nan
    for era, e in d.groupby("era", sort=False):
        typical = IsotonicRegression(increasing=True).fit(e["quali"], e["team_pace"])
        dev = e["team_pace"] - pd.Series(typical.predict(e["quali"]), index=e.index)
        finished = e["outcome"] == "finished"
        out = (e["outcome"] == "out").astype(float)
        d.loc[e.index, "exp_finish"], c_f = _monotone_plus_team(e, e["finish"], finished, dev)
        d.loc[e.index, "exp_out"], c_o = _monotone_plus_team(
            e, out, pd.Series(True, index=e.index), dev, y_min=0, y_max=1
        )
        print(f"  {era}: a team 1 slot slower than typical finishes {c_f:+.2f} places worse "
              f"and retires {c_o * 100:+.1f} pp more")
    d["exp_out"] = d["exp_out"].clip(0, 1)
    return d


def slot_curves(d: pd.DataFrame) -> pd.DataFrame:
    """Slot-only expected finish and P(retire) per era: the curves shown, not the scoring baseline."""
    slots = np.arange(1, MAX_SLOT + 1)
    parts = []
    for era, e in d.groupby("era", sort=False):
        fin = e[e["outcome"] == "finished"]
        exp_finish = IsotonicRegression(increasing=True).fit(fin["quali"], fin["finish"]).predict(slots)
        exp_out = IsotonicRegression(increasing=True, y_min=0, y_max=1).fit(
            e["quali"], (e["outcome"] == "out").astype(float)
        ).predict(slots)
        parts.append(pd.DataFrame({"era": era, "quali": slots, "exp_finish": exp_finish, "exp_out": exp_out}))
    return pd.concat(parts, ignore_index=True)


def group_effects(d: pd.DataFrame, keys, min_starts: int) -> pd.DataFrame:
    """Mean places gained and retirements avoided against the era-slot expectation."""
    d = d.assign(
        gain=d["exp_finish"] - d["finish"],
        out_avoided=(d["exp_out"] - (d["outcome"] == "out")) * 100,
    )
    g = d.groupby(keys)
    gf = d[d["outcome"] == "finished"].groupby(keys)
    tbl = pd.DataFrame(
        {
            "starts": g.size(),
            "finishes": gf.size(),
            "avg_quali": g["quali"].mean(),
            "avg_finish": gf["finish"].mean(),
            "gain": gf["gain"].mean(),
            "gain_se": gf["gain"].std(ddof=1) / np.sqrt(gf.size()),
            "out_avoided_pp": g["out_avoided"].mean(),
            "out_se": g["out_avoided"].std(ddof=1) / np.sqrt(g.size()),
        }
    )
    return tbl[tbl["starts"] >= min_starts].copy()


def wilson(k: pd.Series, n: pd.Series, z: float = Z95) -> tuple[pd.Series, pd.Series]:
    """Wilson score interval: honest at the small n and extreme rates circuits give."""
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def per_race_stickiness(d: pd.DataFrame) -> pd.DataFrame:
    """Per race: rank correlation of qualifying and finishing order, and did pole win."""
    keys = ["year", "race"]
    f = d[d["outcome"] == "finished"].copy()
    f["q_rank"] = f.groupby(keys)["quali"].rank()
    f["f_rank"] = f.groupby(keys)["finish"].rank()
    g = f.groupby(keys)
    out = g[["q_rank", "f_rank"]].apply(lambda x: x["q_rank"].corr(x["f_rank"])).rename("rho").to_frame()
    out["finishers"] = g.size()
    # float, so a race with no pole row is NaN and drops out of count() rather than
    # turning the column to object dtype.
    out["pole_won"] = (
        d[d["quali"] == 1].assign(won=lambda x: x["finish"] == 1).groupby(keys)["won"].max()
    ).astype(float)
    out = out[out["finishers"] >= MIN_FINISHERS_FOR_RHO].reset_index()
    out["era"] = out["year"].map(era_label)
    # Shift every era to the overall mean so an era's looser or tighter running
    # order (refuelling, DRS) is not credited to the circuits that happened to run in it.
    out["rho_adj"] = out["rho"] - out.groupby("era")["rho"].transform("mean") + out["rho"].mean()
    return out


def circuit_table(d: pd.DataFrame, per_race: pd.DataFrame) -> pd.DataFrame:
    g = per_race.groupby("race")
    tbl = pd.DataFrame(
        {
            "races": g.size(),
            "rho_raw": g["rho"].mean(),
            "rho": g["rho_adj"].mean(),
            "rho_se": g["rho_adj"].std(ddof=1) / np.sqrt(g.size()),
            "pole_wins": g["pole_won"].sum(),
            "pole_races": g["pole_won"].count(),
        }
    )
    tbl["pole_win_pct"] = tbl["pole_wins"] / tbl["pole_races"] * 100
    lo, hi = wilson(tbl["pole_wins"], tbl["pole_races"])
    tbl["pole_lo"], tbl["pole_hi"] = lo * 100, hi * 100
    tbl = tbl.join(group_effects(d, "race", 0)[["out_avoided_pp", "out_se"]])
    return tbl[tbl["races"] >= MIN_CIRCUIT_RACES].copy()


# --------------------------------------------------------------------------- plot


def _signed(digits: int):
    """Tick formatter: explicit sign, and a bare 0 at the reference line."""
    return lambda v, _: "0" if abs(v) < 10 ** -(digits + 1) else f"{v:+.{digits}f}"


def dot_panel(ax, y, est, lo, hi, ref, fmt) -> None:
    """One row per group: estimate and 95% interval, blue if the interval excludes ``ref``."""
    # Steps of 2/5/10 only: the default locator picks 0.25, which a one-decimal
    # formatter then mislabels (-0.75 reads "-0.8").
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[2, 5, 10]))
    sig = ((lo > ref) | (hi < ref)).to_numpy()
    colors = np.where(sig, BLUE, MUTED)
    ax.hlines(y, lo, hi, colors=colors, lw=1.6, zorder=2)
    ax.scatter(est, y, s=34, c=colors, zorder=3, edgecolors=SURFACE, linewidths=1.2)
    ax.axvline(ref, color=INK_2, lw=1, zorder=1)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.set_major_formatter(fmt)


def _header(fig, height: float, left: float, title: str, subtitle: str) -> None:
    fig.text(left, 1 - 0.34 / height, title, fontsize=15, fontweight="bold", ha="left")
    fig.text(left, 1 - 0.68 / height, subtitle, fontsize=10.5, color=INK_2, ha="left")


def _emphasis_legend(fig, ref_name: str) -> None:
    handles = [
        plt.Line2D([0], [0], color=BLUE, lw=1.6, marker="o", ms=5, label=f"95% interval excludes {ref_name}"),
        plt.Line2D([0], [0], color=MUTED, lw=1.6, marker="o", ms=5, label=f"includes {ref_name}: not distinguishable"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=2, frameon=False, fontsize=9.5)


def plot_effects(tbl: pd.DataFrame, *, title: str, subtitle: str, left: float, path: Path) -> None:
    t = tbl.sort_values("gain")  # drawn bottom to top, so the biggest gainer is on top
    y = np.arange(len(t))
    height = 1.95 + 0.27 * len(t)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.8, height), sharey=True, gridspec_kw={"wspace": 0.05})
    ci = Z95 * t["gain_se"]
    dot_panel(a1, y, t["gain"], t["gain"] - ci, t["gain"] + ci, 0, _signed(1))
    ci = Z95 * t["out_se"]
    dot_panel(a2, y, t["out_avoided_pp"], t["out_avoided_pp"] - ci, t["out_avoided_pp"] + ci, 0, _signed(0))
    a1.set_yticks(y)
    a1.set_yticklabels(t["label"], fontsize=9.5)
    a1.tick_params(axis="y", colors=INK_2)
    a1.set_xlabel("Places gained vs a typical car from the same slot (right = better)", fontsize=9.5)
    a2.set_xlabel("Retirements avoided vs typical, percentage points (right = better)", fontsize=9.5)
    _header(fig, height, left, title, subtitle)
    _emphasis_legend(fig, "zero")
    fig.subplots_adjust(left=left, right=0.985, top=1 - 1.0 / height, bottom=1.1 / height)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_era_heatmaps(d: pd.DataFrame, path: Path) -> None:
    labels = [e[0] for e in ERAS]
    parts = {}
    for era in labels:
        e = d[d["era"] == era]
        parts[era] = (outcome_matrix(e, "quali"), slot_summary(e, "quali"), e)
    vmax = float(np.ceil(max(p.to_numpy().max() for p, _, _ in parts.values()) * 100 / 5) * 5)

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n + 1.2, 5.5), sharey=True)
    mappable = None
    for ax, era in zip(np.atleast_1d(axes), labels):
        prob, summary, e = parts[era]
        mappable = draw_heatmap(ax, prob, summary, vmax, compact=True, marker_size=14)
        ax.set_title(era, loc="left", fontsize=11, fontweight="bold", pad=19)
        ax.annotate(
            f"pole wins {summary.loc[1, 'win_pct']:.0f}% · {(e['outcome'] == 'out').mean() * 100:.0f}% retire",
            xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
            fontsize=9, color=INK_2, va="bottom", ha="left",
        )
        ax.set_xlabel("Finishing position", fontsize=9.5)
    np.atleast_1d(axes)[0].set_ylabel("Qualifying position")

    first, last = parts[labels[0]][1].loc[1, "win_pct"], parts[labels[-1]][1].loc[1, "win_pct"]
    r0, r1 = (parts[labels[i]][2]["outcome"].eq("out").mean() * 100 for i in (0, -1))
    fig.text(0.015, 0.955, f"Pole wins {first:.0f}% of races in {labels[0]} but {last:.0f}% in {labels[-1]}, "
             f"while retirements fall from {r0:.0f}% to {r1:.0f}%", fontsize=15, fontweight="bold", ha="left")
    fig.text(0.015, 0.915, "Share of cars from each qualifying slot reaching each outcome, by era · "
             "line = average finish (finishers only)", fontsize=10.5, color=INK_2, ha="left")
    cb = fig.colorbar(mappable, ax=list(np.atleast_1d(axes)), fraction=0.02, pad=0.015)
    cb.set_label("% of cars from that qualifying slot", color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9)
    cb.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    fig.subplots_adjust(left=0.05, right=0.92, top=0.80, bottom=0.11, wspace=0.07)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _spread(ys: np.ndarray, gap: float) -> np.ndarray:
    """Nudge direct labels apart so neighbours do not overprint."""
    out = ys.astype(float).copy()
    order = np.argsort(out)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def plot_era_curves(baseline: pd.DataFrame, path: Path) -> None:
    labels = [e[0] for e in ERAS]
    colors = ordinal_colors(len(labels))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    a1.plot([1, MAX_SLOT], [1, MAX_SLOT], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    # y is inverted, so the diagonal descends on screen; sit the label below it,
    # in the empty region of "finished worse than qualified".
    a1.annotate("finish = qualify", xy=(16, 16), xytext=(-2, -9), textcoords="offset points",
                color=MUTED, fontsize=8.5, rotation=-41, rotation_mode="anchor", ha="center", va="top")
    for ax, col, scale in ((a1, "exp_finish", 1), (a2, "exp_out", 100)):
        ends = np.array([baseline[(baseline["era"] == e) & (baseline["quali"] == MAX_SLOT)][col].iloc[0] * scale
                         for e in labels])
        text_y = _spread(ends, 0.8 if col == "exp_finish" else 2.2)
        for era, color, end, ty in zip(labels, colors, ends, text_y):
            b = baseline[baseline["era"] == era]
            ax.plot(b["quali"], b[col] * scale, color=color, lw=2, zorder=3)
            ax.annotate(era, xy=(MAX_SLOT, end), xytext=(MAX_SLOT + 0.7, ty), color=INK_2, fontsize=9.5,
                        va="center", ha="left")
        ax.set_xlim(0.5, MAX_SLOT + 4.2)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlabel("Qualifying position", fontsize=9.5)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    a1.set_ylim(MAX_SLOT + 0.5, 0.5)
    a1.set_yticks([1, 5, 10, 15, 20])
    a1.set_ylabel("Expected finishing position (finishers)", fontsize=9.5)
    a2.set_ylim(0, None)
    a2.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    a2.set_ylabel("Expected probability of retiring", fontsize=9.5)
    handles = [plt.Line2D([0], [0], color=c, lw=2, label=e) for e, c in zip(labels, colors)]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=len(labels), frameon=False,
               fontsize=9.5)
    # Read the headline off the data: attrition promotes everyone who survives, so
    # the back of the grid finished *better* when many more cars retired.
    at = lambda era, col: baseline[(baseline["era"] == era) & (baseline["quali"] == MAX_SLOT)][col].iloc[0]
    fig.text(0.015, 0.955, f"Retirements used to reshuffle the grid: a P{MAX_SLOT} starter finished "
             f"P{at(labels[0], 'exp_finish'):.0f} in {labels[0]}, P{at(labels[-1], 'exp_finish'):.0f} in {labels[-1]}",
             fontsize=15, fontweight="bold", ha="left")
    fig.text(0.015, 0.915, "Expected outcome for a car from each qualifying slot, by era · finishers' position "
             "(left) and chance of retiring (right) · monotone fit to every start", fontsize=10.5, color=INK_2,
             ha="left")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.17, wspace=0.18)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_circuits(tbl: pd.DataFrame, ref_rho: float, ref_pole: float, path: Path) -> None:
    t = tbl.sort_values("rho")  # stickiest on top
    y = np.arange(len(t))
    height = 1.95 + 0.30 * len(t)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, height), sharey=True, gridspec_kw={"wspace": 0.08})
    ci = Z95 * t["rho_se"]
    dot_panel(axes[0], y, t["rho"], t["rho"] - ci, t["rho"] + ci, ref_rho, lambda v, _: f"{v:.2f}")
    dot_panel(axes[1], y, t["pole_win_pct"], t["pole_lo"], t["pole_hi"], ref_pole, lambda v, _: f"{v:.0f}%")
    ci = Z95 * t["out_se"]
    dot_panel(axes[2], y, t["out_avoided_pp"], t["out_avoided_pp"] - ci, t["out_avoided_pp"] + ci, 0, _signed(0))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(t["label"], fontsize=9.5)
    axes[0].tick_params(axis="y", colors=INK_2)
    axes[0].set_xlabel("Grid stickiness: rank correlation of\nqualifying and finish order (right = stickier)", fontsize=9.5)
    axes[1].set_xlabel("Pole-sitter wins\n(line = all circuits)", fontsize=9.5)
    axes[2].set_xlabel("Retirements avoided vs typical,\npercentage points (right = fewer)", fontsize=9.5)
    # Headline only what the intervals support: the extremes of a ranking built on
    # 7-25 races each are mostly noise, and the chart paints them grey.
    ci = Z95 * t["rho_se"]
    sticky = [s for s in t.index[(t["rho"] - ci) > ref_rho]][::-1]
    loose = [s for s in t.index[(t["rho"] + ci) < ref_rho]]

    def gp(slugs: list[str]) -> str:
        names = [s.replace("-", " ").title() for s in slugs]
        return " and ".join(names) if len(names) <= 2 else ", ".join(names[:-1]) + f" and {names[-1]}"

    title = (f"Qualifying order sticks most at the {gp(sticky)} GP{'s' if len(sticky) > 1 else ''}"
             if sticky else "No Grand Prix is clearly stickier than average")
    title += (f", and least at the {gp(loose)} GP{'s' if len(loose) > 1 else ''}" if loose
              else "; none is clearly looser")
    left = 0.17
    _header(fig, height, left, title,
            f"Per Grand Prix, 2000–2024 · blue = interval excludes the all-circuit average "
            f"· 1.0 = finish in qualifying order · † venue changed")
    _emphasis_legend(fig, "the all-circuit line")
    fig.subplots_adjust(left=left, right=0.985, top=1 - 1.0 / height, bottom=1.35 / height)
    fig.savefig(path, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- main


def _span(names: list[str]) -> str:
    """'BAR', 'Sauber/Alfa Romeo' or 'BAR–Brawn': the names a driver raced under, in order."""
    if len(names) <= 2:
        return "/".join(names)
    return f"{names[0]}–{names[-1]}"


def raced_as(d: pd.DataFrame) -> pd.Series:
    """For each (driver, lineage): the constructor names actually raced under, earliest first.

    A pairing is keyed on the organisation, but labelled with the era's own names:
    "J. Button / Mercedes" would be false (he drove BAR, Honda and Brawn, which are
    the same factory that later became Mercedes).
    """
    ordered = d.sort_values("year")
    return ordered.groupby(["driver", "lineage"])["constructor"].agg(lambda s: _span(list(dict.fromkeys(s))))


def _initial_surname(driver: str) -> str:
    parts = driver.split()
    surname = " ".join(parts[-2:]) if parts[-1] in {"Jr.", "Jr"} else parts[-1]
    return f"{parts[0][0]}. {surname}"


def _show(name: str, tbl: pd.DataFrame, col: str = "gain", n: int = 5) -> None:
    cols = ["starts", "avg_quali", "gain", "gain_se", "out_avoided_pp"]
    t = tbl.sort_values(col, ascending=False)[cols].round(2)
    print(f"\n{name}: top {n} and bottom {n} of {len(t)} by {col}")
    print(pd.concat([t.head(n), t.tail(n)]).to_string())


def main() -> int:
    races = load_races()
    print("Team-pace correction to the expectation, per era:")
    d = prepare(races)
    baseline = slot_curves(d)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    print(f"{len(d):,} starts, {d.groupby(['year', 'race']).ngroups} races, eras "
          f"{', '.join(f'{e} ({(d.era == e).sum():,})' for e, _, _ in ERAS)}")

    # -- era
    plot_era_heatmaps(d, FIGURES_DIR / "qualifying_by_era_heatmaps.png")
    plot_era_curves(baseline, FIGURES_DIR / "qualifying_by_era_curves.png")
    era_tbl = pd.concat({e: slot_summary(d[d["era"] == e], "quali") for e, _, _ in ERAS}, names=["era"])
    era_tbl.round(2).to_csv(FIGURES_DIR / "qualifying_by_era.csv")
    print("\nPole win % / retirement % by era:")
    print(pd.DataFrame({
        "pole_win_pct": era_tbl.xs(1, level="quali_slot")["win_pct"],
        "retire_pct": d.groupby("era").apply(lambda e: (e["outcome"] == "out").mean() * 100, include_groups=False),
    }).reindex([e for e, _, _ in ERAS]).round(1).to_string())

    # -- driver
    drv = group_effects(d, "driver", MIN_STARTS["driver"])
    drv["label"] = [f"{n}  ·  {int(s)}" for n, s in zip(drv.index, drv["starts"])]
    _show("Drivers", drv)
    best = drv.sort_values("gain").iloc[-1]
    plot_effects(
        drv, left=0.17, path=FIGURES_DIR / "qualifying_by_driver.png",
        title=f"{drv['gain'].idxmax()} gains the most on Sunday: {best['gain']:+.1f} places over a typical car "
              f"from the same slot",
        subtitle=f"Drivers with {MIN_STARTS['driver']}+ starts, 2000–2024 · number = starts · "
                 f"includes the car, so see constructors and pairings next",
    )
    drv.drop(columns="label").round(3).to_csv(FIGURES_DIR / "qualifying_by_driver.csv")

    # -- constructor
    con = group_effects(d, "lineage", MIN_STARTS["constructor"])
    con["label"] = [f"{n}  ·  {int(s)}" for n, s in zip(con.index, con["starts"])]
    _show("Constructors", con)
    # The check that the team-pace term is doing its job: against a slot-only
    # baseline this correlation was -0.91 (the chart ranked cars, not execution).
    print(f"  check: corr(average qualifying slot, gain) across constructors = "
          f"{con['gain'].corr(con['avg_quali']):+.2f}")
    best = con.sort_values("gain").iloc[-1]
    plot_effects(
        con, left=0.33, path=FIGURES_DIR / "qualifying_by_constructor.png",
        title=f"{short_name(con['gain'].idxmax())} gains the most on Sunday: {best['gain']:+.1f} places over a "
              f"typical car from the same slot",
        subtitle=f"Constructors with {MIN_STARTS['constructor']}+ starts, grouped by organisation across rebrands "
                 f"· number = starts",
    )
    con.drop(columns="label").round(3).to_csv(FIGURES_DIR / "qualifying_by_constructor.csv")

    # -- driver x constructor
    pair = group_effects(d, ["driver", "lineage"], MIN_STARTS["pairing"])
    names = raced_as(d)
    pair["label"] = [f"{_initial_surname(dr)}  ·  {names[(dr, ln)]}  ·  {int(s)}"
                     for (dr, ln), s in zip(pair.index, pair["starts"])]
    _show("Driver-constructor pairings", pair)
    best = pair.sort_values("gain").iloc[-1]
    plot_effects(
        pair, left=0.31, path=FIGURES_DIR / "qualifying_by_pairing.png",
        title=f"{best['label'].split('  ·  ')[0]} at {best['label'].split('  ·  ')[1]} gains the most: "
              f"{best['gain']:+.1f} places over typical",
        subtitle=f"Driver-constructor pairings with {MIN_STARTS['pairing']}+ starts, 2000–2024 · "
                 f"number = starts · same driver, different car",
    )
    pair.drop(columns="label").round(3).to_csv(FIGURES_DIR / "qualifying_by_pairing.csv")

    # -- circuit
    per_race = per_race_stickiness(d)
    cir = circuit_table(d, per_race)
    cir["label"] = [f"{s.replace('-', ' ').title()}{'  †' if s in VENUE_CHANGED else ''}  ·  {int(r)}"
                    for s, r in zip(cir.index, cir["races"])]
    ref_rho = float(per_race["rho_adj"].mean())
    ref_pole = float(per_race["pole_won"].mean() * 100)
    print(f"\nCircuits ({len(cir)} with {MIN_CIRCUIT_RACES}+ races): stickiness vs era-adjusted mean {ref_rho:.3f}; "
          f"pole win rate overall {ref_pole:.1f}%")
    print(cir.sort_values("rho", ascending=False)[
        ["races", "rho_raw", "rho", "rho_se", "pole_win_pct", "out_avoided_pp"]].round(2).to_string())
    plot_circuits(cir, ref_rho, ref_pole, FIGURES_DIR / "qualifying_by_circuit.png")
    cir.drop(columns="label").round(3).to_csv(FIGURES_DIR / "qualifying_by_circuit.csv")

    print(f"\nWrote figures and tables to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
