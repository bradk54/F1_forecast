"""Chart theme and the handful of plot forms this dashboard reuses.

One place decides what a chart looks like, so twelve charts across four pages
read as one instrument rather than twelve.  The rules encoded here, and why:

* **Colour carries identity, never magnitude that a length already carries.**
  A risk ranking is one series in one hue; the bar length is the magnitude.
  Colouring ten bars by ten teams would spend the only free channel restating
  the axis, and ten categorical hues cannot be told apart under colour-vision
  deficiency anyway.
* **Categorical hues are assigned in fixed slot order and never cycled.**  Past
  three simultaneous series in a scatter, or eight anywhere, the answer is
  facets, not a ninth hue.
* **Never two y-axes.**  Two measures at different scales get two charts.
* **Diverging means blue/red across a neutral grey zero**, used only where zero
  genuinely separates better from worse — an ablation delta, a calibration
  residual.  A sequential ramp is one hue, light to dark.
* **Grid and axes are solid hairlines one shade off the surface**, never dashed:
  a dashed rule reads as a threshold, and here the only dashed line on any chart
  *is* a threshold (the base rate, the 45° calibration diagonal).

Palette values are the data-viz reference palette, used unchanged.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

#: Categorical slots in fixed order.  Take them from the front; do not reorder,
#: and do not add a ninth.
SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767")

#: Single-hue ramp for genuine magnitude encoding (heat, density).
SEQUENTIAL_LIGHT = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                    "#256abf", "#184f95", "#0d366b")
SEQUENTIAL_DARK = SEQUENTIAL_LIGHT

THEME = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e8e7e3",
        "axis": "#d6d5d0",
        "series": SERIES_LIGHT,
        "sequential": SEQUENTIAL_LIGHT,
        "positive": "#2a78d6",
        "negative": "#e34948",
        "neutral": "#b5b4af",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#302f2c",
        "axis": "#3d3c38",
        "series": SERIES_DARK,
        "sequential": SEQUENTIAL_DARK,
        "positive": "#3987e5",
        "negative": "#e66767",
        "neutral": "#6b6a65",
    },
}

FONT = ("ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "Helvetica, Arial, sans-serif")


def mode() -> str:
    """The viewer's theme, so charts are stepped for the surface behind them.

    Streamlit exposes this on ``st.context`` in recent versions and not at all
    in older ones; a dashboard should not crash over a colour lookup, so an
    unavailable theme falls back to light.
    """
    try:
        theme_type = getattr(getattr(st, "context", None), "theme", None)
        if theme_type is not None and getattr(theme_type, "type", None) == "dark":
            return "dark"
    except Exception:  # noqa: BLE001 - a theme probe must never break a page
        pass
    try:
        if st.get_option("theme.base") == "dark":
            return "dark"
    except Exception:  # noqa: BLE001
        pass
    return "light"


def palette() -> dict:
    return THEME[mode()]


# --------------------------------------------------------------------------- #
# Shared layout
# --------------------------------------------------------------------------- #


def style(
    fig: go.Figure,
    *,
    height: int = 340,
    xtitle: str = "",
    ytitle: str = "",
    legend: bool = False,
    yzero: bool = False,
    margin_right: int = 8,
    margin_top: int | None = None,
) -> go.Figure:
    """Apply the house layout: recessive chrome, room for the axis band, no dual scale.

    ``height`` sizes the whole container including the x-axis labels, which is
    the usual cause of a chart growing its own tiny scrollbar.  ``margin_right``
    and ``margin_top`` exist so a reference label can live *outside* the plot
    rather than on top of the data.

    The title is only touched when there is one: styling a title that was never
    set makes Plotly render the string "undefined" above the plot.
    """
    colors = palette()
    has_title = bool(fig.layout.title.text)
    fig.update_layout(
        height=height,
        margin=dict(
            l=8, r=margin_right,
            t=margin_top if margin_top is not None else (28 if has_title else 8),
            b=8,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=colors["muted"]),
        hoverlabel=dict(font_family=FONT, font_size=12),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color=colors["muted"]), bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="closest",
    )
    axis = dict(
        gridcolor=colors["grid"], griddash="solid", gridwidth=1,
        linecolor=colors["axis"], zerolinecolor=colors["axis"],
        tickfont=dict(color=colors["muted"], size=11),
        title=dict(font=dict(color=colors["muted"], size=11)),
    )
    fig.update_xaxes(**axis, showgrid=False, title_text=xtitle)
    fig.update_yaxes(**axis, showgrid=True, title_text=ytitle,
                     rangemode="tozero" if yzero else "normal")
    if has_title:
        fig.update_layout(
            title=dict(font=dict(size=13, color=colors["text"]), x=0, xanchor="left")
        )
    return fig


def show(fig: go.Figure) -> None:
    """Render, with the mode bar off — it is chart junk on a local dashboard."""
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _reference(
    fig: go.Figure,
    *,
    value: float,
    label: str,
    axis: str,
    height: int,
    xtitle: str = "",
    ytitle: str = "",
    legend: bool = False,
    yzero: bool = False,
) -> go.Figure:
    """Draw a threshold rule and put its label *outside* the plot.

    Plotly's own annotation positions all sit inside the axes, which on a full
    chart means the label lands on the data — the one label collision that shows
    up on every chart with a base-rate line.  A horizontal rule labels itself in
    the right margin, a vertical rule above the plot, and the margin is reserved
    here rather than by the caller so the two cannot get out of step.

    Dashed is reserved for exactly this: on these charts the only dashed line is
    a threshold, never a grid line and never a series.
    """
    colors = palette()
    if axis == "y":
        pad = min(150, max(56, 6 * len(label) + 12)) if label else 8
        style(fig, height=height, xtitle=xtitle, ytitle=ytitle, legend=legend,
              yzero=yzero, margin_right=pad)
        fig.add_hline(y=value, line=dict(color=colors["muted"], width=1, dash="dash"))
        if label:
            fig.add_annotation(
                text=label, xref="paper", x=1.0, xanchor="left", xshift=6,
                y=value, yanchor="middle", showarrow=False,
                font=dict(color=colors["muted"], size=11),
            )
    else:
        style(fig, height=height, xtitle=xtitle, ytitle=ytitle, legend=legend,
              yzero=yzero, margin_top=26 if label else None)
        fig.add_vline(x=value, line=dict(color=colors["muted"], width=1, dash="dash"))
        if label:
            fig.add_annotation(
                text=label, yref="paper", y=1.0, yanchor="bottom", yshift=4,
                x=value, xanchor="center", showarrow=False,
                font=dict(color=colors["muted"], size=11),
            )
    return fig


# --------------------------------------------------------------------------- #
# Forms
# --------------------------------------------------------------------------- #


def ranked_bars(
    frame: pd.DataFrame,
    *,
    label: str,
    value: str,
    hover: Sequence[str] = (),
    reference: float | None = None,
    reference_label: str = "",
    highlight: int = 0,
    height: int = 520,
    xtitle: str = "",
) -> go.Figure:
    """Horizontal bars, biggest at the top, one series in one hue.

    ``highlight`` emphasises the leading N rows by muting the rest, which is the
    correct way to say "these are the ones that matter" — the alternative,
    giving every bar its own hue, encodes identity that the axis already carries.
    """
    colors = palette()
    ordered = frame.sort_values(value, ascending=True)
    n = len(ordered)
    if highlight > 0:
        fills = [
            colors["series"][0] if i >= n - highlight else colors["neutral"]
            for i in range(n)
        ]
    else:
        fills = [colors["series"][0]] * n

    custom = ordered[list(hover)].to_numpy() if hover else None
    template = "<b>%{y}</b><br>" + f"{xtitle or value}: " + "%{x:.3f}"
    for i, name in enumerate(hover):
        template += f"<br>{name}: " + "%{customdata[" + str(i) + "]}"
    template += "<extra></extra>"

    fig = go.Figure(
        go.Bar(
            x=ordered[value], y=ordered[label], orientation="h",
            marker=dict(color=fills, cornerradius=4),
            customdata=custom, hovertemplate=template,
        )
    )
    if reference is not None:
        _reference(fig, value=reference, label=reference_label, axis="x",
                   height=height, xtitle=xtitle)
    else:
        style(fig, height=height, xtitle=xtitle)
    fig.update_layout(bargap=0.35)
    return fig


def diverging_bars(
    frame: pd.DataFrame,
    *,
    label: str,
    value: str,
    height: int = 380,
    xtitle: str = "",
) -> go.Figure:
    """Horizontal bars across a neutral zero, for signed deltas.

    Blue above zero, red below, grey at it — the only place a two-hue split is
    right is when zero genuinely separates better from worse.
    """
    colors = palette()
    ordered = frame.sort_values(value)
    fills = [
        colors["negative"] if v < 0 else colors["positive"] if v > 0
        else colors["neutral"]
        for v in ordered[value]
    ]
    fig = go.Figure(
        go.Bar(
            x=ordered[value], y=ordered[label], orientation="h",
            marker=dict(color=fills, cornerradius=4),
            hovertemplate="<b>%{y}</b><br>" + f"{xtitle or value}: "
                          + "%{x:+.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color=colors["axis"], width=1))
    style(fig, height=height, xtitle=xtitle)
    fig.update_layout(bargap=0.35)
    return fig


def series_lines(
    frame: pd.DataFrame,
    *,
    x: str,
    series: dict[str, str],
    height: int = 340,
    xtitle: str = "",
    ytitle: str = "",
    reference: float | None = None,
    reference_label: str = "",
    markers: bool = False,
) -> go.Figure:
    """One line per named series, hues taken from the front of the slot order.

    ``series`` maps a column to its display name.  Two series get a legend; that
    is not optional, because identity must never be colour alone.
    """
    colors = palette()
    fig = go.Figure()
    for i, (column, name) in enumerate(series.items()):
        fig.add_trace(
            go.Scatter(
                x=frame[x], y=frame[column], name=name, mode="lines+markers" if markers else "lines",
                line=dict(color=colors["series"][i % len(colors["series"])], width=2),
                marker=dict(size=8, line=dict(width=2, color=colors["surface"])),
                connectgaps=False,
                hovertemplate=f"<b>{name}</b><br>%{{x}}<br>%{{y:.3f}}<extra></extra>",
            )
        )
    if reference is not None:
        _reference(fig, value=reference, label=reference_label, axis="y",
                   height=height, xtitle=xtitle, ytitle=ytitle,
                   legend=len(series) > 1)
    else:
        style(fig, height=height, xtitle=xtitle, ytitle=ytitle,
              legend=len(series) > 1)
    return fig


def scatter(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    text: str | None = None,
    hover: Sequence[str] = (),
    height: int = 380,
    xtitle: str = "",
    ytitle: str = "",
    color_by: str | None = None,
    color_names: Sequence[str] = (),
    diagonal: bool = False,
    size: str | None = None,
) -> go.Figure:
    """Points, capped at three colour groups.

    Three is not arbitrary: it is the number of slots from this palette that
    clear the colour-vision gates when *every* pair can appear side by side, as
    they do in a scatter.  A fourth group facets instead.
    """
    colors = palette()
    fig = go.Figure()

    def marker_size(block: pd.DataFrame):
        if size is None:
            return 9
        values = pd.to_numeric(block[size], errors="coerce").fillna(0)
        span = values.max() - values.min()
        return 9 + 12 * ((values - values.min()) / span if span else 0)

    groups = (
        [(name, frame.loc[frame[color_by] == name]) for name in color_names]
        if color_by else [(None, frame)]
    )
    for i, (name, block) in enumerate(groups):
        if block.empty:
            continue
        custom = block[list(hover)].to_numpy() if hover else None
        template = f"{xtitle or x}: %{{x:.3f}}<br>{ytitle or y}: %{{y:.3f}}"
        if text:
            template = "<b>%{text}</b><br>" + template
        for j, column in enumerate(hover):
            template += f"<br>{column}: " + "%{customdata[" + str(j) + "]}"
        fig.add_trace(
            go.Scatter(
                x=block[x], y=block[y], mode="markers",
                name=name or "", text=block[text] if text else None,
                customdata=custom,
                marker=dict(
                    size=marker_size(block),
                    color=colors["series"][i % 3],
                    line=dict(width=2, color=colors["surface"]),
                    opacity=0.9,
                ),
                hovertemplate=template + "<extra></extra>",
            )
        )
    if diagonal:
        low = float(min(frame[x].min(), frame[y].min()))
        high = float(max(frame[x].max(), frame[y].max()))
        fig.add_trace(
            go.Scatter(
                x=[low, high], y=[low, high], mode="lines", showlegend=False,
                line=dict(color=colors["muted"], width=1, dash="dash"),
                hoverinfo="skip",
            )
        )
    style(fig, height=height, xtitle=xtitle, ytitle=ytitle,
          legend=bool(color_by))
    return fig


def bars(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    height: int = 320,
    xtitle: str = "",
    ytitle: str = "",
    hover: Sequence[str] = (),
    reference: float | None = None,
    reference_label: str = "",
) -> go.Figure:
    """Vertical bars, one series, one hue — for an ordered category like a season."""
    colors = palette()
    custom = frame[list(hover)].to_numpy() if hover else None
    template = "<b>%{x}</b><br>" + f"{ytitle or y}: " + "%{y:.3f}"
    for i, name in enumerate(hover):
        template += f"<br>{name}: " + "%{customdata[" + str(i) + "]}"
    fig = go.Figure(
        go.Bar(
            x=frame[x], y=frame[y],
            marker=dict(color=colors["series"][0], cornerradius=4),
            customdata=custom, hovertemplate=template + "<extra></extra>",
        )
    )
    if reference is not None:
        _reference(fig, value=reference, label=reference_label, axis="y",
                   height=height, xtitle=xtitle, ytitle=ytitle, yzero=True)
    else:
        style(fig, height=height, xtitle=xtitle, ytitle=ytitle, yzero=True)
    fig.update_layout(bargap=0.3)
    return fig


def histogram_by_outcome(
    frame: pd.DataFrame, column: str, target: str = "dnf", *, height: int = 320
) -> go.Figure:
    """Two overlaid densities: the feature among finishers and among retirements.

    Two series, two slots, a legend — the standard "does this feature separate
    the classes at all" look.
    """
    colors = palette()
    fig = go.Figure()
    for i, (value, name) in enumerate(((0, "finished"), (1, "retired"))):
        block = pd.to_numeric(
            frame.loc[frame[target] == value, column], errors="coerce"
        ).dropna()
        if block.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=block, name=name, histnorm="probability density",
                marker=dict(color=colors["series"][i], line=dict(width=0)),
                opacity=0.65, nbinsx=30,
            )
        )
    fig.update_layout(barmode="overlay")
    style(fig, height=height, xtitle=column, ytitle="density", legend=True)
    return fig


def empty(message: str, *, height: int = 200) -> go.Figure:
    """A placeholder that says why there is nothing, rather than an empty box."""
    colors = palette()
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False,
        font=dict(color=colors["muted"], size=13), x=0.5, y=0.5, xref="paper",
        yref="paper",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    style(fig, height=height)
    return fig
