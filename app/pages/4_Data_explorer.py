"""Browse the modelling table, the feature registry, and the circuit profiles.

Three questions this answers that a notebook cell answers slowly:

* **Where does attrition actually live** — which teams, drivers and circuits
  carry it, and how much of an apparent effect is just a small denominator.
* **Does a feature separate the classes at all**, before it is worth arguing
  about in an ablation.
* **Is the registry telling the truth** — a feature can be declared, selected,
  and still arrive 100% null, which the registry alone cannot show and
  ``audit_coverage`` only reports at build time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import charts, services  # noqa: E402
from app.common import metric_row, page_setup, require_dataset, sidebar_status  # noqa: E402

page_setup("Data explorer", icon="🔎")

version = services.data_version()
if not require_dataset():
    st.stop()

dataset = services.load_dataset(version)
manifest = services.manifest_or_none()
sidebar_status(dataset, manifest)

st.title("Data explorer")

# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #

seasons = sorted(dataset["Year"].dropna().unique().astype(int))
controls = st.columns([3, 3, 3])
with controls[0]:
    span = st.select_slider(
        "Seasons", options=seasons, value=(seasons[0], seasons[-1]),
    )
with controls[1]:
    teams = sorted(dataset["TeamId"].dropna().astype(str).unique())
    picked_teams = st.multiselect("Teams", teams, default=[])
with controls[2]:
    eras = sorted(dataset["regulation_era"].dropna().astype(str).unique())
    picked_eras = st.multiselect("Regulation eras", eras, default=[])

frame = dataset.loc[dataset["Year"].between(span[0], span[1])]
if picked_teams:
    frame = frame.loc[frame["TeamId"].astype(str).isin(picked_teams)]
if picked_eras:
    frame = frame.loc[frame["regulation_era"].astype(str).isin(picked_eras)]

if frame.empty:
    st.warning("No rows match those filters.")
    st.stop()

races = frame[["Year", "RoundNumber"]].drop_duplicates().shape[0]
metric_row(
    [
        ("Driver-races", f"{len(frame):,}", f"{races} races"),
        ("Retirements", f"{int(frame['dnf'].sum()):,}", f"{frame['dnf'].mean():.1%}"),
        ("Drivers", f"{frame['DriverId'].nunique()}", "distinct"),
        ("Circuits", f"{frame['circuit_key'].nunique()}", "distinct"),
    ]
)

tab_where, tab_features, tab_circuits, tab_registry, tab_rows = st.tabs(
    ["Where attrition lives", "Features", "Circuits", "Registry", "Rows"]
)

# --------------------------------------------------------------------------- #
# Where attrition lives
# --------------------------------------------------------------------------- #

with tab_where:
    st.caption(
        "Rates are shown with the count behind them. A 40% retirement rate over "
        "five starts is a small denominator, not a finding — the minimum-starts "
        "control is there to stop the chart being a list of reserve drivers."
    )
    minimum = st.slider("Minimum starts", 1, 120, 30, key="min_starts")

    by = st.radio(
        "Group by", ["Team", "Driver", "Circuit", "Season"],
        horizontal=True, key="group_by",
    )
    key = {"Team": "TeamId", "Driver": "DriverId",
           "Circuit": "circuit_short_name", "Season": "Year"}[by]
    if key not in frame.columns:
        key = "TeamId"

    grouped = (
        frame.groupby(key, observed=True)
        .agg(starts=("dnf", "size"), retirements=("dnf", "sum"), rate=("dnf", "mean"))
        .reset_index()
    )
    grouped = grouped.loc[grouped["starts"] >= minimum]
    grouped[key] = grouped[key].astype(str)

    if grouped.empty:
        st.warning(f"Nothing has at least {minimum} starts under these filters.")
    else:
        left, right = st.columns([3, 2])
        with left:
            top = grouped.nlargest(min(25, len(grouped)), "rate")
            fig = charts.ranked_bars(
                top, label=key, value="rate",
                hover=("starts", "retirements"), highlight=3,
                reference=float(frame["dnf"].mean()),
                reference_label="overall rate",
                xtitle="retirement rate", height=max(320, 24 * len(top)),
            )
            fig.update_xaxes(tickformat=".0%")
            charts.show(fig)
        with right:
            display = grouped.sort_values("rate", ascending=False).copy()
            display["rate"] = display["rate"].map("{:.1%}".format)
            st.dataframe(
                display, hide_index=True, width="stretch",
                height=max(320, 24 * min(25, len(grouped))),
            )

    st.divider()
    st.subheader("Race by race")
    st.caption(
        "Every race in the filtered set. The rolling line is what the training "
        "window is chasing — and the reason a 40-race window was chosen over an "
        "expanding one going into 2026."
    )
    per_race = services.race_attrition(frame)
    per_race["rolling"] = per_race["dnf_rate"].rolling(10, min_periods=3).mean()
    fig = charts.series_lines(
        per_race, x="race_date",
        series={"dnf_rate": "per race", "rolling": "10-race mean"},
        ytitle="retirement rate", height=320,
    )
    fig.data[0].update(mode="markers", marker=dict(size=6, line=dict(width=0)), opacity=0.4)
    fig.update_yaxes(tickformat=".0%")
    charts.show(fig)

# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #

with tab_features:
    stage = st.radio("Stage", services.STAGES, index=1, horizontal=True, key="fx_stage")
    available = services.stage_features(frame, stage)
    numeric = [
        c for c in available
        if pd.api.types.is_numeric_dtype(frame[c]) and frame[c].notna().any()
    ]
    if not numeric:
        st.warning("No numeric features available at this stage.")
    else:
        default = "grid_position" if "grid_position" in numeric else numeric[0]
        feature = st.selectbox(
            "Feature", numeric, index=numeric.index(default), key="fx_feature"
        )
        description = services.describe_feature(feature)
        if description:
            st.caption(description)

        column = pd.to_numeric(frame[feature], errors="coerce")
        metric_row(
            [
                ("Coverage", f"{column.notna().mean():.1%}",
                 f"{int(column.isna().sum()):,} null"),
                ("Median", f"{column.median():.3f}" if column.notna().any() else "—",
                 f"IQR {column.quantile(.25):.3f}–{column.quantile(.75):.3f}"
                 if column.notna().any() else ""),
                ("Distinct", f"{column.nunique():,}", ""),
                ("Corr. with dnf",
                 f"{column.corr(frame['dnf'].astype(float)):+.3f}"
                 if column.notna().sum() > 2 else "—",
                 "Pearson, linear only"),
            ]
        )

        left, right = st.columns(2)
        with left:
            charts.show(charts.histogram_by_outcome(frame, feature))
            st.caption(
                "Two densities. Where they overlap completely, the feature "
                "carries nothing on its own — which does not mean it carries "
                "nothing in combination."
            )
        with right:
            binned = frame.dropna(subset=[feature]).copy()
            try:
                binned["bucket"] = pd.qcut(
                    pd.to_numeric(binned[feature], errors="coerce"),
                    q=8, duplicates="drop",
                )
            except ValueError:
                binned["bucket"] = pd.cut(
                    pd.to_numeric(binned[feature], errors="coerce"), bins=8
                )
            rate = (
                binned.groupby("bucket", observed=True)
                .agg(rate=("dnf", "mean"), n=("dnf", "size"),
                     centre=(feature, "median"))
                .reset_index()
            )
            rate["bucket"] = rate["bucket"].astype(str)
            fig = charts.bars(
                rate, x="bucket", y="rate", hover=("n", "centre"),
                ytitle="retirement rate", xtitle=f"{feature} (octile)",
                height=320, reference=float(frame["dnf"].mean()),
                reference_label="overall",
            )
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(tickangle=-40)
            charts.show(fig)
            st.caption("Retirement rate by octile — monotone here means a usable signal.")

# --------------------------------------------------------------------------- #
# Circuits
# --------------------------------------------------------------------------- #

with tab_circuits:
    st.caption(
        "The circuit is measured, not named: a one-hot per venue learns nothing "
        "that transfers to a track the model has not seen. The two stress "
        "indices are hypotheses with hand-chosen weights, not findings — the "
        "ablation on the **Lab** page is where they get falsified."
    )
    indices = ["track_speed_index", "mechanical_stress_index", "incident_exposure_index"]
    if not all(c in frame.columns for c in indices):
        st.warning("Circuit profile columns are not present in this dataset.")
    else:
        profile = (
            frame.groupby(["circuit_short_name"], observed=True)
            .agg(
                starts=("dnf", "size"),
                dnf_rate=("dnf", "mean"),
                speed=("track_speed_index", "mean"),
                mechanical=("mechanical_stress_index", "mean"),
                incident=("incident_exposure_index", "mean"),
                corners=("corners_per_km", "mean"),
                lap_km=("lap_length_m", "mean"),
            )
            .reset_index()
        )
        profile["lap_km"] = profile["lap_km"] / 1000
        profile = profile.loc[profile["starts"] >= 20]

        axis = st.selectbox(
            "Index", ["mechanical", "incident", "speed", "corners"], index=0,
        )
        left, right = st.columns([3, 2])
        with left:
            fig = charts.scatter(
                profile, x=axis, y="dnf_rate", text="circuit_short_name",
                hover=("starts", "lap_km"), size="starts",
                xtitle=f"{axis} index", ytitle="retirement rate", height=420,
            )
            fig.update_yaxes(tickformat=".0%")
            charts.show(fig)
            st.caption(
                "Marker size is the number of starts, so a circuit with one "
                "visit does not read as loudly as one with eight."
            )
        with right:
            display = profile.sort_values("dnf_rate", ascending=False).round(3)
            display["dnf_rate"] = display["dnf_rate"].map("{:.1%}".format)
            st.dataframe(
                display, hide_index=True, width="stretch", height=420,
            )

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

with tab_registry:
    st.caption(
        "The registry is authoritative: every feature is declared with the stage "
        "at which it becomes knowable, and `audit_coverage` fails the build for "
        "anything in the dataset but not registered. **Coverage** is the column "
        "to scan — a registered feature arriving 100% null is a build problem "
        "the registry alone cannot show."
    )
    table = services.registry_table(version, dataset)

    filters = st.columns([2, 2, 3])
    with filters[0]:
        stage_filter = st.multiselect(
            "Stage", sorted(table["stage"].unique()), default=[]
        )
    with filters[1]:
        only_problems = st.checkbox("Only problems", value=False)
    with filters[2]:
        search = st.text_input("Search", placeholder="grid, circuit, rate ...")

    view = table.copy()
    if stage_filter:
        view = view.loc[view["stage"].isin(stage_filter)]
    if only_problems:
        view = view.loc[(~view["in_dataset"]) | (view["coverage"] < 0.5)]
    if search:
        needle = search.lower()
        view = view.loc[
            view["feature"].str.lower().str.contains(needle)
            | view["description"].str.lower().str.contains(needle)
        ]

    absent = int((~table["in_dataset"]).sum())
    if absent:
        st.warning(
            f"{absent} registered feature(s) are not in the dataset at all. "
            "`feature_columns(available=...)` skips them, so the model runs "
            "without them silently."
        )

    st.dataframe(
        view[["feature", "stage", "kind", "in_dataset", "coverage", "distinct",
              "description"]],
        hide_index=True, width="stretch", height=520,
        column_config={
            "coverage": st.column_config.ProgressColumn(
                "coverage", format="%.0f%%", min_value=0.0, max_value=1.0,
            )
        },
    )
    st.caption(f"{len(view)} of {len(table)} registered features.")

# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #

with tab_rows:
    st.caption("The filtered modelling table, newest first.")
    identity = [
        c for c in ["Year", "RoundNumber", "EventName", "DriverId", "TeamId",
                    "grid_position", "Position", "Status", "dnf", "dnf_cause"]
        if c in frame.columns
    ]
    extra = st.multiselect(
        "Extra columns", [c for c in frame.columns if c not in identity], default=[]
    )
    view = frame[identity + extra].sort_values(
        ["Year", "RoundNumber"], ascending=False
    )
    st.dataframe(view, hide_index=True, width="stretch", height=520)
    st.download_button(
        "Download this view as CSV",
        view.to_csv(index=False).encode(),
        file_name="dnf_dataset_view.csv", mime="text/csv",
    )
