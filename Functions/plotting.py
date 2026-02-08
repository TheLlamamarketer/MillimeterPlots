from __future__ import annotations
import types
import warnings
"""
Plotting utilities focused on crisp aesthetics, reproducibility, and flexible input formats.
Key upgrades in this revision:
- **Rock-solid contrast**: palette never collides with background; multi-axis contrast fix
  (V, then S, then hue) with a hard fallback to black/white. Optional per-artist
  outline stroke for guaranteed readability.
- **Auto theme**: `style="auto"` chooses light/dark from background luminance.
- **Scientific/engineering ticks**: helpers for SI/Eng units, power limits, and
  auto-labeling of units without duplicating existing labels.
- **Group-aware colors** with stable per-series variations (no reliance on `id()`),
  plus color-blind-safe palette option.
- Clean inputs: `(x, y)`, `(x, y, label)`, dict (compat with your keys), or `DatasetSpec`.
- Returns `(fig, ax)`; optional export to PDF/PNG/SVG.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import hashlib
import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.ticker import AutoMinorLocator, EngFormatter, ScalarFormatter, LogLocator, LogFormatter, LogFormatterMathtext
import matplotlib.colors as mcolors
from matplotlib import patheffects as pe
from matplotlib.axes import Axes
import colorsys

# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """Specifications for a dataset to be plotted."""

    x: Sequence[float] | np.ndarray | float
    y: Sequence[float] | np.ndarray | float

    # labeling / grouping
    label: str | None = None
    color: str | None = None            # explicit hex overrides palette
    color_group: str | None = None      # datasets with same group share a base color
    key: str | None = None              # stable identifier for color hashing

    # drawing
    marker: str = "."                      # "o", "s", "D", "X", "P", ".", "v", "^", "<", ">"
    line: str = "None"                     # "-", "--", ":", "None"
    linewidth: float = 1.0
    markersize: float = 24
    alpha: float = 0.95
    zorder: int = 3
    
    # plot type (for future extension, e.g. "bar", "scatter", etc.). For now, all are "line" (markers+lines).
    plot_style: str = "linear"  # "linear" (default), "bar", "scatter", etc.
    barwidth: float = 0.8      # only for plot_style="bar"

    # error bars (sym or asym). Each can be scalar, array, or (lower, upper)
    yerr: float | Sequence[float] | Tuple[Sequence[float], Sequence[float]] | Tuple[float, float] | None = None
    xerr: float | Sequence[float] | Tuple[Sequence[float], Sequence[float]] | Tuple[float, float] | None = None

    # fits & intervals
    fit_y: Sequence[float] | types.LambdaType | None  = None         
    fit_x: Sequence[float] | None = None
    fit_line: str = "-"
    fit_label: bool = True
    fit_error_lines: List[Tuple[Sequence[float] | None, Sequence[float] | None]] | None = None
    fit_color: str | None = None
    fit_color_group: str | None = None
    
    # Axlines 
    axlines: List[Tuple[float, str]] | None = None  # list of (position, line_style) to draw vertical/horizontal lines, eg. [(2.5, "h"), (3.4, "v")]
    axlines_label: List[str] | None = None
    axlines_show_label: bool | List[bool] | None = None  # whether to show text labels next to axlines (bool for all, or list per-line)
    axlines_line: str = "--"
    axlines_intervals: List[Tuple[float, float]] | None = None
    axlines_color: str | None = None
    axlines_color_group: str | None = None

    # General text annotations: list of dicts with keys: 'x', 'y', 'text', optional: 'offset', 'fontsize', 'color', 'arrow', 'bbox'
    text_annotations: List[Dict[str, Any]] | None = None

    # confidence intervals: list of (lower, upper) arrays matching x (or fit_x)
    confidence: List[Tuple[Sequence[float] | None, Sequence[float] | None]] | None = None
    confidence_label: bool = True

    # misc
    aggregate_duplicates: bool = False      # turn duplicated (x,y) samples into larger markers


# -----------------------------------------------------------------------------
# Color & contrast helpers
# -----------------------------------------------------------------------------

_DEF_OKABE_ITO = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9",
    "#CC79A7", "#F0E442", "#000000"
]


def _stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) & 0x7FFFFFFF


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)


def _relative_luminance(rgb: Tuple[float, float, float]) -> float:
    r, g, b = np.array(rgb, dtype=float)
    r_lin, g_lin, b_lin = _srgb_to_linear(np.array([r, g, b]))
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def _contrast_ratio(rgb1: Tuple[float, float, float], rgb2: Tuple[float, float, float]) -> float:
    L1 = _relative_luminance(rgb1)
    L2 = _relative_luminance(rgb2)
    Lh, Ll = (L1, L2) if L1 >= L2 else (L2, L1)
    return (Lh + 0.05) / (Ll + 0.05)

def _ensure_contrast(
    rgb: Tuple[float, float, float],
    bg_rgb: Tuple[float, float, float],
    *,
    min_contrast: float = 1.5,
    max_h_div: float = 0.1          # avoid colors too close to bg hue
) -> Tuple[float, float, float]:
    """Adjust color to meet `min_contrast` vs background"""
    cr_og = _contrast_ratio(rgb, bg_rgb)
    if cr_og >= min_contrast:
        return rgb

    def hue_dist(h1, h2):
        d = abs(h1 - h2) % 1.0
        return min(d, 1.0 - d)
    
    cand_list = []

    bg_L = _relative_luminance(bg_rgb)
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    for step in np.linspace(0.02, 0.8, 40):
        v_try = np.clip(v - step, 0, 1) if bg_L > 0.5 else np.clip(v + step, 0, 1)
        cand = colorsys.hsv_to_rgb(h, s, v_try)
        cr = _contrast_ratio(cand, bg_rgb)
        if cr >= min_contrast: 
            return cand
        else: cand_list.append([cand, cr])
    
    # Background & original in HSV
    hb, sb, vb = colorsys.rgb_to_hsv(*bg_rgb)
        
    if hue_dist(hb, h) < 0.05:
        for i in np.linspace(0, max_h_div, 20):
            cand = colorsys.hsv_to_rgb(h + i, s, v_try)
            cr = _contrast_ratio(cand, bg_rgb)
            if cr >= min_contrast: return cand
            else: cand_list.append([cand, cr])
    cand_list.sort(key=lambda t:t[1], reverse=True)
    return cand_list[0][0] if cr_og <cand_list[0][1] else rgb  # best effort

def generate_palette(
    n: int,
    *,
    bg_hex: str = "#FFFFFF",
    seed: int | None = 203,
    saturation: float = 0.72,
    value: float = 0.82,
    method: str = "golden",
    variation: float = 0.07,
    min_contrast: float = 1.5,
    base_palette: list[str] | None = None
) -> list[str]:
    """Create n distinct colors that contrast with background.

    method: "golden" (hue steps by golden angle), "linspace" (even spread), or "okabe".
    If `base_palette` has >= n, it is truncated and contrast-enforced.
    """
    bg_rgb = mcolors.to_rgb(bg_hex)

    def enforce(col_hex: str) -> str:
        c = mcolors.to_rgb(col_hex)
        c = _ensure_contrast(c, bg_rgb, min_contrast=min_contrast)
        return mcolors.to_hex(c)

    if base_palette and len(base_palette) >= n:
        return [enforce(c) for c in base_palette[:n]]

    if method == "okabe":
        cols = _DEF_OKABE_ITO * ((n + len(_DEF_OKABE_ITO) - 1) // len(_DEF_OKABE_ITO))
        return [enforce(c) for c in cols[:n]]

    rng = np.random.default_rng(seed)
    base = rng.random()

    if method == "golden":
        step = (math.sqrt(5) - 1) / 2.0  # ~0.618...
        hues = (base + step * np.arange(n)) % 1.0
    else:  # linspace
        hues = (base + np.linspace(0, 1, n, endpoint=False)) % 1.0

    hues = (hues + rng.uniform(-variation, variation, size=n)) % 1.0

    cols = [colorsys.hsv_to_rgb(h, saturation, value) for h in hues]
    cols = [_ensure_contrast(c, bg_rgb, min_contrast=min_contrast) for c in cols]
    return [mcolors.to_hex(c) for c in cols]


def colors_from_groups(
    datasets: Sequence["DatasetSpec" | Mapping[str, Any]],
    *,
    color_seed: int | None = 203,
    bg_hex: str = "#FFFFFF",
    hue_variation_within_group: float = 0.04,
    palette_method: str = "golden",
    min_contrast: float = 1.5,
) -> Dict[str, str]:
    """Assign a color to each dataset keyed by its stable dataset key."""
    norm: List[DatasetSpec] = [normalize_dataset(d) for d in datasets]

    groups: Dict[str, List[DatasetSpec]] = {}
    singles: List[DatasetSpec] = []

    for d in norm:
        if d.color is not None:
            continue
        if d.color_group:
            groups.setdefault(d.color_group, []).append(d)
        else:
            singles.append(d)

    n_groups = len(groups)
    n_singles = len(singles)
    base_palette = generate_palette(
        n_groups + n_singles,
        bg_hex=bg_hex,
        seed=color_seed,
        method=palette_method,
        min_contrast=min_contrast,
    )

    out: Dict[str, str] = {}
    idx = 0

    for g, items in groups.items():
        base_color = base_palette[idx]
        idx += 1
        # Small variations per member (hue wobble only; keep S/V)
        base_rgb = mcolors.to_rgb(base_color)
        h, s, v = colorsys.rgb_to_hsv(*base_rgb)

        # stable RNG per group
        rng = np.random.default_rng((_stable_int(g) ^ (color_seed or 0)) & 0xFFFFFFFF)
        for d in items:
            if d.color is None:
                hj = (h + rng.uniform(-hue_variation_within_group, hue_variation_within_group)) % 1.0
                c = colorsys.hsv_to_rgb(hj, s, v)
                c = _ensure_contrast(c, mcolors.to_rgb(bg_hex), min_contrast=min_contrast)
                out[_ds_key(d)] = mcolors.to_hex(c)

    for d in singles:
        out[_ds_key(d)] = base_palette[idx]
        idx += 1

    for d in norm:  # explicit colors win
        if d.color is not None:
            out[_ds_key(d)] = mcolors.to_hex(d.color)

    return out


# -----------------------------------------------------------------------------
# Input normalization
# -----------------------------------------------------------------------------
def normalize_dataset(
    d: DatasetSpec | Mapping[str, Any] | Tuple[Sequence[float], Sequence[float]] | Tuple[Sequence[float], Sequence[float], str]
) -> DatasetSpec:
    if isinstance(d, DatasetSpec):
        return d

    if isinstance(d, tuple):
        if len(d) == 2:
            x, y = d
            return DatasetSpec(x=x, y=y)
        elif len(d) == 3:
            x, y, label = d
            return DatasetSpec(x=x, y=y, label=str(label))
        else:
            raise ValueError("Tuple dataset must be (x, y) or (x, y, label)")

    if isinstance(d, Mapping):
        x = d.get("x") or d.get("xdata")
        y = d.get("y") or d.get("ydata")
        if x is None or y is None:
            raise ValueError("Mapping dataset must include 'x' and 'y' (or 'xdata' and 'ydata')")
        return DatasetSpec(
            x=x,
            y=y, 
            label=d.get("label"),
            color=d.get("color"),
            color_group=d.get("color_group"),
            key=d.get("key"),
            marker=d.get("marker", "."),
            line=d.get("line", "None"),
            linewidth=d.get("linewidth", 1.4),
            markersize=d.get("markersize", 24),
            alpha=d.get("alpha", 0.95),
            zorder=d.get("zorder", 3),
            yerr=d.get("yerr") or d.get("y_error"),
            xerr=d.get("xerr") or d.get("x_error"),
            fit_y=d.get("fit"),
            fit_x=d.get("fit_xdata"),
            fit_line=d.get("fit_line", "-"),
            fit_color=d.get("fit_color"),
            fit_color_group=d.get("fit_color_group"),
            fit_label=d.get("fit_label", True),
            fit_error_lines=d.get("fit_error_lines"),
            axlines=d.get("axlines"),
            axlines_label=d.get("axlines_label"),
            axlines_show_label=d.get("axlines_show_label"),
            axlines_line=d.get("axlines_line", "--"),
            axlines_color=d.get("axlines_color"),
            axlines_color_group=d.get("axlines_color_group"),
            axlines_intervals=d.get("axlines_intervals"),
            text_annotations=d.get("text_annotations"),
            confidence=d.get("confidence"),
            confidence_label=d.get("confidence_label", True),
            aggregate_duplicates=d.get("aggregate_duplicates", False),
        )

    raise TypeError("Unsupported dataset type. Use DatasetSpec, dict, or (x,y).")


def _ds_key(d: DatasetSpec) -> str:
    if d.key:
        return d.key
    base = d.label or "series"
    cg = d.color_group or "nogroup"
    # Stable pseudo-key
    return f"{base}|{cg}|{_stable_int(str(len(np.atleast_1d(d.x))))}"


# -----------------------------------------------------------------------------
# Theme & grid helpers
# -----------------------------------------------------------------------------

FONT_RC = {
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
}

_LIGHT_RC = {
    **FONT_RC,
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "axes.edgecolor": "#A0A0A0",
    "axes.grid": True,
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.8,
    "grid.alpha": 1.0,
    "xtick.color": "#404040",
    "ytick.color": "#404040",
    "axes.labelcolor": "#202020",
    "text.color": "#202020",
}

_DARK_RC = {
    **FONT_RC,
    "axes.facecolor": "#0F1115",
    "figure.facecolor": "#0F1115",
    "axes.edgecolor": "#5E6A7D",
    "axes.grid": True,
    "grid.color": "#2A3140",
    "grid.linewidth": 0.8,
    "grid.alpha": 1.0,
    "xtick.color": "#D8DEE9",
    "ytick.color": "#D8DEE9",
    "axes.labelcolor": "#E5E9F0",
    "text.color": "#ECEFF4",
}

def _apply_bg(ax: Axes, bg_hex: str):
    ax.set_facecolor(bg_hex)
    ax.figure.set_facecolor(bg_hex)
    ax.figure.set_facecolor(bg_hex)


def _auto_style_from_bg(bg_hex: str) -> str:
    L = _relative_luminance(mcolors.to_rgb(bg_hex))
    return "dark" if L < 0.5 else "light"


# -----------------------------------------------------------------------------
# SI/Eng formatting helpers
# -----------------------------------------------------------------------------

def apply_si_format(
    ax: Axes,
    *,
    xunit: str | None = None,
    yunit: str | None = None,
    x_eng: bool = True,
    y_eng: bool = True,
    sci_limits: Tuple[int, int] = (-3, 3),
    put_unit_in_label: bool = True,
):
    """Apply scientific/engineering formatters with optional units.

    - If `x_eng`/`y_eng` is True, use EngFormatter (k, m, µ, …).
    - Else use ScalarFormatter with power limits `sci_limits`.
    - If the axis label lacks a trailing unit like "[V]", append it.
    """
    def _ensure_unit(lbl: str | None, unit: str | None) -> str | None:
        if not put_unit_in_label or not unit:
            return lbl
        if lbl is None or lbl == "":
            return f"[{unit}]"
        if "[" in lbl and "]" in lbl:
            return lbl  # assume already present
        return f"{lbl} [{unit}]"

    if x_eng:
        ax.xaxis.set_major_formatter(EngFormatter(unit=xunit or ""))
    else:
        xf = ScalarFormatter(useMathText=True)
        xf.set_powerlimits(sci_limits)
        ax.xaxis.set_major_formatter(xf)

    if y_eng:
        ax.yaxis.set_major_formatter(EngFormatter(unit=yunit or ""))
    else:
        yf = ScalarFormatter(useMathText=True)
        yf.set_powerlimits(sci_limits)
        ax.yaxis.set_major_formatter(yf)

    x_lbl = _ensure_unit(ax.get_xlabel(), xunit)
    if x_lbl is not None:
        ax.set_xlabel(x_lbl)
    y_lbl = _ensure_unit(ax.get_ylabel(), yunit)
    if y_lbl is not None:
        ax.set_ylabel(y_lbl)


# Helper function to dispatch plot based on style
def _plot_series_data(
    ax: Axes,
    s: DatasetSpec,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    oc: str,
    outline: bool,
    outline_width: float,
) -> None:
    """Plot a single series based on plot_style.
    
    Handles the actual plotting (errorbar/scatter/line vs bar) and returns nothing.
    This is called from within the loop over specs in plot_data.
    """
    xerr_lower, xerr_upper = _normalize_err(s.xerr, x)
    yerr_lower, yerr_upper = _normalize_err(s.yerr, y)

    if s.plot_style == "bar":
        # Bar plot: x is positions (0, 1, 2, ...) or categorical labels
        yerr = None 
        if s.yerr is not None:
            yerr = [np.abs(yerr_lower), np.abs(yerr_upper)]
        
        bars = ax.bar(
            x, y,
            color=color,
            alpha=s.alpha,
            zorder=s.zorder + 1,
            width=s.barwidth,
            label=s.label,
            edgecolor=(oc if outline else None),
            linewidth=(outline_width if outline else None),
            yerr=yerr,
            capsize=5 if s.yerr is not None else 0,
            error_kw={"elinewidth": 0.9, "capthick": 0.9}
        )
        
        if outline:
            for bar in bars:
                bar.set_path_effects([
                    pe.Stroke(linewidth=outline_width, foreground=oc),
                    pe.Normal(),
                ])
    else:
        # "linear" (default): scatter + line plot
        if s.xerr is not None or s.yerr is not None:
            eb = ax.errorbar(
                x, y,
                xerr=None if s.xerr is None else [np.abs(xerr_lower), np.abs(xerr_upper)],
                yerr=None if s.yerr is None else [np.abs(yerr_lower), np.abs(yerr_upper)],
                fmt=s.marker if s.marker != "None" else "o",
                markeredgecolor=(oc if outline else None) if not s.marker == "x" else None,
                color=color, ecolor=color, elinewidth=0.9, capsize=3, capthick=0.9, linestyle=s.line,
                markersize=max(4, s.markersize / 3), alpha=s.alpha, zorder=s.zorder+1,
                label=s.label)
            if outline:
                for art in eb.lines + eb.caplines:  # type: ignore[attr-defined]
                    art.set_path_effects([
                        pe.Stroke(linewidth=art.get_linewidth()+outline_width, foreground=oc), pe.Normal()
                    ])
        else:
            if s.marker != "None":
                sc = ax.scatter(x, y, color=color, marker=s.marker, s=s.markersize, alpha=s.alpha, zorder=s.zorder+1,
                                label=s.label, edgecolors=(oc if outline else None), linewidths=(outline_width if outline else None))
            if s.line != "None":
                line_main, = ax.plot(x, y, linestyle=s.line, color=color, linewidth=s.linewidth, zorder=s.zorder+1,
                                    label=None if s.marker != "None" else s.label)
                if outline:
                    line_main.set_path_effects([
                        pe.Stroke(linewidth=line_main.get_linewidth()+outline_width, foreground=oc), pe.Normal()
                    ])


# -----------------------------------------------------------------------------
# Main plotting
# -----------------------------------------------------------------------------

def plot_data(
    datasets: Sequence[DatasetSpec | Mapping[str, Any] | Tuple[Any, Any] | Tuple[Any, Any, str]],
    *,
    filename: str | None = None,
    color_seed: int | None = 203,
    title: str | None = None,
    subtitle: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend_position: str | None = "best",
    width: float = 20,
    height: float = 20,
    dpi: int = 200,
    bg_hex: str = "#FFFFFF",
    palette_method: str = "golden",
    min_contrast: float = 1.5,
    log_scale: Tuple[int, int] | None = None,  # (x_base, y_base)
    xticks: Tuple[float, float] | None = None,   # (step, offset)
    yticks: Tuple[float, float] | None = None,   
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    tight: bool = True,
    transparent: bool = False,
    export_svg: bool = False,
    export_png: bool = False,
    # outlines for guaranteed visibility
    outline: bool = False,
    outline_width: float = 1.5,
    outline_color: str | None = None,
    # SI/Eng formatting
    xunit: str | None = None,
    yunit: str | None = None,
    x_eng: bool = True,
    y_eng: bool = True,
    sci_limits: Tuple[int, int] = (-3, 3),
    put_unit_in_label: bool = True,
    plot: bool | str = False
) -> Tuple[Figure, Axes]:
    """Comprehensive 2D plotting function with multiple customization options.
    
    Parameters
    ----------
    datasets : Sequence of DatasetSpec, dict, or (x,y) tuples
        List of datasets to plot.
    filename : str, optional
        Output filename (PDF/PNG/SVG based on extension).
    color_seed : int, optional
        Seed for color generation.
    title : str, optional
        Plot title.
    subtitle : str, optional
        Plot subtitle.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    legend_position : str or None, optional
        Matplotlib legend position string or None for no legend.
    width : float, optional
        Figure width in cm.
    height : float, optional
        Figure height in cm.
    dpi : int, optional
        Figure DPI.
    bg_hex : str, optional
        Background color hex.
    palette_method : str, optional
        "golden", "linspace", or "okabe" for color generation.
    min_contrast : float, optional
        Minimum contrast ratio between colors and background.
    log_scale : tuple of int or None, optional
        (x_base, y_base) for log scaling, or None.
    xticks : tuple of float or None, optional
        (step, offset) for dense grid lines, or None.
    yticks : tuple of float or None, optional
        (step, offset) for dense grid lines, or None.
    xlim : tuple of float or None, optional
        X-axis limits.
    ylim : tuple of float or None, optional
        Y-axis limits.
    tight : bool, optional
        Use tight layout.
    transparent : bool, optional
        Save figure with transparent background.
    export_svg : bool, optional
        Also export to SVG alongside filename.
    export_png : bool, optional
        Also export to PNG alongside filename.
    outline : bool, optional
        Whether to draw outlines for visibility.
    outline_width : float, optional
        Width of outlines.
    outline_color : str or None, optional
        Color of outlines.
    xunit : str or None, optional
        Unit for x axis.
    yunit : str or None, optional
        Unit for y axis.
    x_eng : bool, optional
        Whether to use engineering notation for x axis.
    y_eng : bool, optional
        Whether to use engineering notation for y axis.
    sci_limits : tuple of int, optional
        Scientific notation limits.
    put_unit_in_label : bool, optional
        Whether to include units in axis labels.
    plot : bool or str, optional
        Whether to display the plot or return the figure and axes using (plot=True/False or "figure")

    Returns
    -----------
    fig : .Figure
    ax : `~matplotlib.axes.Axes`

    
    """
    # Normalize datasets and filter out empties early
    if not isinstance(datasets, (list, tuple)):
        datasets = [datasets]
    specs = [normalize_dataset(d) for d in datasets]
    specs = [s for s in specs if s.x is not None and s.y is not None]

    # Colors
    color_map = colors_from_groups(
        specs,
        color_seed=color_seed,
        bg_hex=bg_hex,
        palette_method=palette_method,
        min_contrast=min_contrast,
    )
    if specs is None or len(specs) == 0:
        pass

    # Styling
    rc = {**(_LIGHT_RC if _auto_style_from_bg(bg_hex) == "light" else _DARK_RC)}
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(width/2.54, height/2.54), dpi=dpi, constrained_layout=True)
        _apply_bg(ax, bg_hex)
        
        
        if log_scale is not None:
            x_log, y_log = log_scale
            if x_log is not None:
                if x_log <= 1:
                    raise ValueError("x_log base must be > 1 for log scale")
                ax.set_xscale("log", base=x_log)
                # Ensure locator/formatter are set for arbitrary bases (e.g. np.e)
                ax.xaxis.set_major_locator(LogLocator(base=x_log))
                # For arbitrary bases prefer a mathtext formatter (e.g. show $e^{2}$)
                try:
                    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=x_log, labelOnlyBase=False))
                except Exception:
                    ax.xaxis.set_major_formatter(LogFormatter(base=x_log, labelOnlyBase=False))
            if y_log is not None:
                if y_log <= 1:
                    raise ValueError("y_log base must be > 1 for log scale")
                ax.set_yscale("log", base=y_log)
                # Ensure locator/formatter are set for arbitrary bases (e.g. np.e)
                ax.yaxis.set_major_locator(LogLocator(base=y_log))
                try:
                    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=y_log, labelOnlyBase=False))
                except Exception:
                    ax.yaxis.set_major_formatter(LogFormatter(base=y_log, labelOnlyBase=False))
        
        is_x_log = ax.get_xscale() == "log"
        is_y_log = ax.get_yscale() == "log"

        if xticks is not None and yticks is not None and xlim is not None and ylim is not None and not (is_x_log or is_y_log):
            _draw_dense_grid(ax, xlim, ylim, width, height, bg_hex, xticks, yticks)
        else:
            gray = mcolors.to_rgb("#a0a0a0")
            ax.grid(True, which="major", color=_ensure_contrast(gray, mcolors.to_rgb(bg_hex)))
            ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.8,
                    color=_ensure_contrast(gray, mcolors.to_rgb(bg_hex)))

            if not is_x_log:
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            # for log axes, let Matplotlib’s default LogLocator/Formatter handle minors
            if not is_y_log:
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        # Outline color defaults to high-contrast vs background
        oc_rgb = (1.0, 1.0, 1.0) if _contrast_ratio((1.0, 1.0, 1.0), mcolors.to_rgb(bg_hex)) >= _contrast_ratio((0.0, 0.0, 0.0), mcolors.to_rgb(bg_hex)) else (0.0, 0.0, 0.0)
        oc = outline_color or mcolors.to_hex(oc_rgb)

        # Plot series
        for s in specs:
            x = np.atleast_1d(np.asarray(s.x, dtype=float))
            y = np.atleast_1d(np.asarray(s.y, dtype=float))
            if x.size == 0 or y.size == 0:
                continue
            if x.shape != y.shape:
                y = y.reshape(x.shape)

            color = s.color or color_map[_ds_key(s)]

            # Confidence bands (behind everything)
            if s.confidence is not None:
                for key in s.confidence.keys() if isinstance(s.confidence, dict) else range(len(s.confidence)):
                    lo, hi = s.confidence[key]
                    if lo is None or hi is None:
                        continue
                    xci = np.asarray(s.fit_x if s.fit_x is not None else x, dtype=float)
                    lo = np.asarray(lo, dtype=float)
                    hi = np.asarray(hi, dtype=float)
                    alpha = max(0.1, 0.5 - 0.08 * key)
                    ax.fill_between(xci, lo, hi, facecolor=color, edgecolor=None, 
                                    alpha=alpha, zorder=2, label=s.label + f" {key}σ Confidence Interval" if s.confidence_label and s.label else None)
                    
            # Axlines
            if s.axlines is not None:
                # Normalize colors, labels and intervals so behavior is predictable
                n_lines = len(s.axlines)

                # colors: single string -> use for all; iterable -> per-line, fallback to series color
                if s.axlines_color is None:
                    al_colors = [color] * n_lines
                elif isinstance(s.axlines_color, str):
                    al_colors = [s.axlines_color] * n_lines
                else:
                    al_colors = list(s.axlines_color)
                    if len(al_colors) < n_lines:
                        al_colors += [color] * (n_lines - len(al_colors))

                # labels: single string -> show once (first line only), iterable -> per-line
                if s.axlines_label is None:
                    al_labels = [None] * n_lines
                elif isinstance(s.axlines_label, str):
                    al_labels = [s.axlines_label] + [None] * (n_lines - 1)
                else:
                    al_labels = list(s.axlines_label)
                    if len(al_labels) < n_lines:
                        al_labels += [None] * (n_lines - len(al_labels))

                # show_label: controls whether to draw text annotation next to the line
                if s.axlines_show_label is None:
                    al_show_label = [False] * n_lines
                elif isinstance(s.axlines_show_label, bool):
                    al_show_label = [s.axlines_show_label] * n_lines
                else:
                    al_show_label = list(s.axlines_show_label)
                    if len(al_show_label) < n_lines:
                        al_show_label += [False] * (n_lines - len(al_show_label))

                # intervals (optional ranges): align length, default None
                if s.axlines_intervals is None:
                    al_intervals = [None] * n_lines
                elif isinstance(s.axlines_intervals, (list, tuple)):
                    if len(s.axlines_intervals) == 2:
                        al_intervals = [s.axlines_intervals] * n_lines
                    elif len(s.axlines_intervals) < n_lines:
                        al_intervals += [None] * (n_lines - len(al_intervals))
                    else:
                        al_intervals = list(s.axlines_intervals)

                for i in range(n_lines):
                    pos = s.axlines[i]
                    lbl = al_labels[i]
                    show_lbl = al_show_label[i]
                    inter = al_intervals[i]
                    al_color = al_colors[i]
                    al_line = s.axlines_line[i] if isinstance(s.axlines_line, (list, tuple)) and len(s.axlines_line) > i else s.axlines_line

                    if pos is None:
                        continue

                    # pos can be (value, orientation) or simple scalar
                    if isinstance(pos, (tuple, list)) and len(pos) == 2:
                        val, orientation = pos
                    else:
                        val, orientation = pos, "h"

                    # normalize orientation
                    o = str(orientation).lower()

                    # determine plotting bounds if intervals are not provided
                    if inter is not None and isinstance(inter, (list, tuple)) and len(inter) == 2:
                        low, high = inter
                    else:
                        low, high = None, None

                    try:
                        if o.startswith("h") or o.startswith("-"):
                            xmin = low if low is not None else (xlim[0] if xlim is not None else x.min())
                            xmax = high if high is not None else (xlim[1] if xlim is not None else x.max())
                            al = ax.hlines(y=float(val), xmin=xmin, xmax=xmax, linestyle=al_line, color=al_color, linewidth=1.0, zorder=s.zorder,
                                            label=lbl)
                        elif o.startswith("v") or o.startswith("|") or o.startswith("x"):
                            ymin = low if low is not None else (ylim[0] if ylim is not None else y.min())
                            ymax = high if high is not None else (ylim[1] if ylim is not None else y.max())
                            al = ax.vlines(x=float(val), ymin=ymin, ymax=ymax, linestyle=al_line, color=al_color, linewidth=1.0, zorder=s.zorder,
                                            label=lbl)
                        else:
                            # unknown orientation, skip
                            continue
                    except Exception:
                        # if something goes wrong (e.g., cannot convert val), skip this line
                        continue

                    if outline:
                        # different artists have different attribute access; handle gracefully
                        try:
                            al.set_path_effects([
                                pe.Stroke(linewidth=getattr(al, 'get_linewidth', lambda: 1.0)() + outline_width, foreground=oc), pe.Normal()
                            ])
                        except Exception:
                            pass
                    # If axlines_show_label is True for this line and a label exists, place it next to the drawn line
                    if lbl and show_lbl:
                        try:
                            # Use the interval bounds for text placement (not full axis limits)
                            if o.startswith("h") or o.startswith("-"):
                                text_xmin = low if low is not None else (xlim[0] if xlim is not None else x.min())
                                text_xmax = high if high is not None else (xlim[1] if xlim is not None else x.max())
                                text_bounds = (text_xmin, text_xmax)
                                cur_ylim = ax.get_ylim()
                                _place_text_near_axline(ax, orientation=o, val=float(val), text=lbl,
                                                        color=al_color, x_bounds=text_bounds, y_bounds=cur_ylim,
                                                        outline=outline, outline_color=oc)
                            else:  # vertical
                                text_ymin = low if low is not None else (ylim[0] if ylim is not None else y.min())
                                text_ymax = high if high is not None else (ylim[1] if ylim is not None else y.max())
                                text_bounds = (text_ymin, text_ymax)
                                cur_xlim = ax.get_xlim()
                                _place_text_near_axline(ax, orientation=o, val=float(val), text=lbl,
                                                        color=al_color, x_bounds=cur_xlim, y_bounds=text_bounds,
                                                        outline=outline, outline_color=oc)
                        except Exception:
                            # best-effort only; do not fail plotting on annotation errors
                            pass

            # General text annotations
            if s.text_annotations:
                for ann in s.text_annotations:
                    try:
                        ann_x = ann.get('x')
                        ann_y = ann.get('y')
                        ann_text = ann.get('text')
                        if ann_x is None or ann_y is None or ann_text is None:
                            continue
                        ann_offset = ann.get('offset', (6, 6))
                        ann_fontsize = ann.get('fontsize', 8)
                        ann_color = ann.get('color', color)
                        ann_arrow = ann.get('arrow', False)
                        ann_bbox = ann.get('bbox')
                        annotate_point(ax, ann_x, ann_y, ann_text, offset=ann_offset,
                                      fontsize=ann_fontsize, color=ann_color,
                                      arrow=ann_arrow, bbox=ann_bbox)
                    except Exception:
                        # best-effort: skip problematic annotations
                        pass

            # Fit lines
            if s.fit_y is not None or isinstance(s.fit_y, types.LambdaType):
                xf = np.asarray(s.fit_x if s.fit_x is not None else x, dtype=float)
                if isinstance(s.fit_y, types.LambdaType):
                    yf = s.fit_y(xf)
                else:
                    yf = s.fit_y  # assume array-like
                yf = np.asarray(yf, dtype=float)
                label_fit = f"{s.label} Fit" if (s.label and s.fit_label) else None
                color_fit = s.fit_color or (color_map[_ds_key(s)] if s.fit_color_group is None else color)
                line_fit, = ax.plot(xf, yf, linestyle=s.fit_line, color=color_fit, linewidth=max(1.2, s.linewidth),
                                    label=label_fit, zorder=s.zorder+2)
                if outline:
                    line_fit.set_path_effects([
                        pe.Stroke(linewidth=line_fit.get_linewidth()+outline_width, foreground=oc),
                        pe.Normal(),
                    ])

            # Fit error lines
            if s.fit_error_lines is not None:
                xf = np.asarray(s.fit_x if s.fit_x is not None else x, dtype=float)
                for (lo, hi) in s.fit_error_lines:
                    if lo is not None:
                        le, = ax.plot(xf, np.asarray(lo, dtype=float), linestyle="--", color=color, linewidth=1.0, zorder=s.zorder)
                        if outline:
                            le.set_path_effects([
                                pe.Stroke(linewidth=le.get_linewidth()+outline_width, foreground=oc), pe.Normal()
                            ])
                    if hi is not None:
                        he, = ax.plot(xf, np.asarray(hi, dtype=float), linestyle="--", color=color, linewidth=1.0, zorder=s.zorder)
                        if outline:
                            he.set_path_effects([
                                pe.Stroke(linewidth=he.get_linewidth()+outline_width, foreground=oc), pe.Normal()
                            ])

            if s.aggregate_duplicates:
                # counts per (x,y)
                pts, counts = np.unique(np.c_[x, y], axis=0, return_counts=True)
                x, y = pts[:, 0], pts[:, 1]
                s.markersize = s.markersize + 87 * (np.log(counts) / np.log1p(counts).max())

            # Dispatch to appropriate plotting function based on plot_style
            _plot_series_data(ax, s, x, y, color, oc, outline, outline_width)

        # Labels & title
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if subtitle:
            ax.text(0.01, 1.02, subtitle, transform=ax.transAxes, fontsize=10, alpha=0.8)

        # Limits
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)

        '''
        # Legend (deduplicate labels)
        if legend_position:
            handles, labels = ax.get_legend_handles_labels()
            dedup: Dict[str, Any] = {}
            for h, l in zip(handles, labels):
                if l and l not in dedup:
                    dedup[l] = h
            if dedup:
                ncol = legend_ncol or (1 if len(dedup) < 5 else 2)
                ax.legend(dedup.values(), dedup.keys(), loc=legend_position, frameon=False, ncol=ncol)'''

        handles, labels = ax.get_legend_handles_labels()
        if any(labels) and legend_position:
            ax.legend(loc=legend_position)

        # SI/Eng formatting at the end (tick locs are established)
        if (xunit is not None or yunit is not None or not (x_eng and y_eng)):
            apply_si_format(
                ax,
                xunit=None if is_x_log else xunit,
                yunit=None if is_y_log else yunit,
                x_eng=False if is_x_log else x_eng,
                y_eng=False if is_y_log else y_eng,
                sci_limits=sci_limits,
                put_unit_in_label=put_unit_in_label,
            )

        if tight:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fig.tight_layout()
                
        # Save/Export
        if filename:
            fig.savefig(filename, bbox_inches="tight", dpi=dpi, transparent=transparent)
        if export_png and filename:
            fig.savefig(_with_ext(filename, ".png"), bbox_inches="tight", dpi=max(200, dpi), transparent=transparent)
        if export_svg and filename:
            fig.savefig(_with_ext(filename, ".svg"), bbox_inches="tight", dpi=dpi, transparent=transparent)
            
        if plot == 'figure':
            return fig, ax
        elif plot:
            plt.show()
        else:
            plt.close()

        


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _with_ext(path: str, new_ext: str) -> str:
    import os
    base, _ = os.path.splitext(path)
    return base + new_ext


def _normalize_err(err: Any, base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if err is None:
        return np.zeros_like(base), np.zeros_like(base)
    if isinstance(err, (int, float)):
        e = np.full_like(base, float(err))
        return e, e
    if isinstance(err, (list, tuple)) and len(err) == 2:
        lo, hi = err
        lo = np.full_like(base, float(lo)) if isinstance(lo, (int, float)) else np.asarray(lo, dtype=float)
        hi = np.full_like(base, float(hi)) if isinstance(hi, (int, float)) else np.asarray(hi, dtype=float)
        return lo, hi
    e = np.asarray(err, dtype=float)
    return e, e


def annotate_point(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    *,
    offset: Tuple[float, float] = (6, 6),  # offset in points (x, y)
    fontsize: float = 8,
    color: str | None = None,
    arrow: bool = False,
    bbox: dict | None = None,
    **kwargs,
) -> None:
    """Annotate a point (x, y) in data coordinates with text offset in points.

    Parameters:
    - ax: target Axes
    - x, y: data coordinates of the point
    - text: label to show
    - offset: tuple in points (x_offset, y_offset)
    - arrow: whether to draw a small arrow from the text to the point
    - bbox: dictionary for matplotlib textbox props (facecolor, alpha, boxstyle)
    - kwargs: passed to `ax.annotate` (e.g., ha, va)
    """
    if color is None:
        color = kwargs.pop("color", None)
    if bbox is None:
        bbox = dict(boxstyle="round,pad=0.2", fc=(1, 1, 1, 0.85), ec="none")
    arrowprops = dict(arrowstyle="-", linewidth=0.6, color=color) if arrow else None
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(offset[0], offset[1]),
        textcoords="offset points",
        fontsize=fontsize,
        color=color,
        bbox=bbox,
        arrowprops=arrowprops,
        **kwargs,
    )


def add_side_text(
    ax: Axes,
    text: str,
    *,
    side: str = "right",
    xpad: float = 0.01,
    y: float | None = None,
    fontsize: float = 9,
    color: str | None = None,
    bbox: dict | None = None,
) -> None:
    """Place text in the figure margins using axes fraction coordinates.

    side: 'right'|'left'|'top'|'bottom'
    xpad: fraction of axes width to pad from the chosen side
    y: fraction of axes height for vertical placement (None -> center)
    """
    if bbox is None:
        bbox = dict(boxstyle="round,pad=0.2", fc=(1, 1, 1, 0.0), ec="none")
    if color is None:
        color = None
    y = 0.5 if y is None else y
    if side == "right":
        ax.text(1.0 - xpad, y, text, transform=ax.transAxes, fontsize=fontsize, va="center", ha="right", color=color, bbox=bbox)
    elif side == "left":
        ax.text(xpad, y, text, transform=ax.transAxes, fontsize=fontsize, va="center", ha="left", color=color, bbox=bbox)
    elif side == "top":
        ax.text(0.5, 1.0 - xpad, text, transform=ax.transAxes, fontsize=fontsize, va="top", ha="center", color=color, bbox=bbox)
    elif side == "bottom":
        ax.text(0.5, xpad, text, transform=ax.transAxes, fontsize=fontsize, va="bottom", ha="center", color=color, bbox=bbox)


def _place_text_near_axline(
    ax: Axes,
    *,
    orientation: str,
    val: float,
    text: str,
    color: str | None = None,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    outline: bool = False,
    outline_color: str | None = None,
    fontsize: float = 8,
) -> None:
    """Place a short label next to a horizontal or vertical axline.

    This does a a best-effort placement inside the axes area so labels remain visible.
    """
    o = str(orientation).lower()
    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    x_span = max(1e-12, xmax - xmin)
    y_span = max(1e-12, ymax - ymin)
    color = color or "#000000"

    if o.startswith("h") or o.startswith("-"):
        # place at the right end, slightly inset
        x_text = xmax - 0.015 * x_span
        y_text = val
        ha = "right"
        va = "center"
    else:
        # vertical: place at the top, slightly inset
        x_text = val
        y_text = ymax - 0.02 * y_span
        ha = "center"
        va = "top"

    txt = ax.text(x_text, y_text, text, color=color, fontsize=fontsize, ha=ha, va=va, zorder=1000,
                  bbox=dict(boxstyle="round,pad=0.2", fc=(1,1,1,0.6), ec="none"))
    if outline and outline_color:
        try:
            txt.set_path_effects([pe.Stroke(linewidth=1.5, foreground=outline_color), pe.Normal()])
        except Exception:
            pass


def _draw_dense_grid(
    ax: Axes,
    xlim: Tuple[float, float], ylim: Tuple[float, float],
    width: float, height: float,
    bg_hex: str,
    xticks: Tuple[float, float], yticks: Tuple[float, float],
):
    xmin, xmax = xlim
    ymin, ymax = ylim

    aspect_ratio = (xmax - xmin) / max(1e-12, (ymax - ymin)) * (height / max(1e-12, width))
    ax.set_aspect(aspect_ratio, adjustable='box')

    xstep, xoff = xticks
    ystep, yoff = yticks

    xs_main = np.arange(xmin + xoff, xmax + 1e-9, xstep)
    ys_main = np.arange(ymin + yoff, ymax + 1e-9, ystep)

    def _dense(axis_min, axis_max, px_per_unit: float):
        span = axis_max - axis_min
        return (
            np.arange(axis_min, axis_max, span / (px_per_unit * 10)),
            np.arange(axis_min, axis_max, span / (px_per_unit * 5)),
        )

    xs_ter, xs_sec = _dense(xmin, xmax, width)
    ys_ter, ys_sec = _dense(ymin, ymax, height)

    lines = []
    def _add_vlines(xs):
        for x in xs:
            lines.append([(x, ymin), (x, ymax)])
    def _add_hlines(ys):
        for y in ys:
            lines.append([(xmin, y), (xmax, y)])

    _add_vlines(xs_ter); _add_hlines(ys_ter)
    col_ter = LineCollection(lines, colors=[bg_hex], linewidths=0.2, zorder=0)
    ax.add_collection(col_ter)

    lines = []
    _add_vlines(xs_sec); _add_hlines(ys_sec)
    col_sec = LineCollection(lines, colors=[bg_hex], linewidths=0.5, zorder=0)
    ax.add_collection(col_sec)

    lines = []
    _add_vlines(xs_main); _add_hlines(ys_main)
    col_main = LineCollection(lines, colors=[bg_hex], linewidths=0.8, zorder=0)
    ax.add_collection(col_main)

    ax.set_xticks(xs_main)
    ax.set_yticks(ys_main)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    for spine in ax.spines.values():
        spine.set_edgecolor(bg_hex)
        spine.set_zorder(2)


# -----------------------------------------------------------------------------
# Seed visualization
# -----------------------------------------------------------------------------

def plot_color_seeds(
    seed_range: Tuple[int, int] = (100, 110), num_groups: int = 4,
    bg_hex: str = "#FFFFFF", palette_method: str = "golden"
):
    """Visualize palette changes across seeds for group-level colors."""
    seeds = range(seed_range[0], seed_range[1] + 1)


    with mpl.rc_context({**(_LIGHT_RC if _auto_style_from_bg(bg_hex) == "light" else _DARK_RC)}):
        fig, ax = plt.subplots(figsize=(max(6, len(list(seeds)) * 0.4), num_groups * 0.5), constrained_layout=True)

        for col, seed in enumerate(seeds):
            cols = generate_palette(num_groups, bg_hex=bg_hex, seed=seed, method=palette_method)
            for row in range(num_groups):
                ax.add_patch(Rectangle((col, row), 0.8, 0.9, color=cols[row]))

        _apply_bg(ax,bg_hex )
        ax.set_xticks(np.arange(len(list(seeds))) + 0.5)
        ax.set_xticklabels([str(s) for s in seeds], rotation=45)
        ax.set_yticks(np.arange(num_groups) + 0.5)
        ax.set_yticklabels([f"Group {i+1}" for i in range(num_groups)])
        ax.set_xlim(0, len(list(seeds)))
        ax.set_ylim(0, num_groups)
        ax.invert_yaxis()
        ax.set_frame_on(False)
        ax.grid(False)
        fig.savefig("color_seeds.png", bbox_inches="tight", transparent=False, dpi=150)
        plt.show()



def test_plotting():
    x = np.linspace(0, 10, 600)
    x1 = np.linspace(0, 10, 600)

    y1 = 0.2 * (x-5)**2 + 0.1 * np.random.randn(x.size)

    for i, yi in enumerate(y1):
        if np.abs(yi - y1[i-1]) < 0.1:
            yi = y1[i-1]
            xi = x1[i-1]
        else: xi = x1[i]
        y1[i] = yi
        x1[i] = xi

    s1 = DatasetSpec(x=x1, y=y1, label="Data A", color_group="run", marker=".",  aggregate_duplicates=True)
    s2 = DatasetSpec(x=x, y=np.cos(x) + 0.6*np.random.randn(x.size), label="Data B", color_group="go", fit_y=lambda x_: np.cos(x_), fit_line="-")

    # Add more random data and plots
    y3 = np.sin(x) + 0.3 * np.random.randn(x.size)
    s3 = DatasetSpec(x=x, y=y3, label="Data C", color_group="sin", marker="s", color=None)

    y4 = np.log1p(x) + 0.2 * np.random.randn(x.size)
    s4 = DatasetSpec(x=x, y=y4, label="Data D", color_group="log", marker="^", color=None)

    y5 = np.exp(-0.3 * x) + 0.1 * np.random.randn(x.size)
    s5 = DatasetSpec(x=x, y=y5, label="Data E", color_group="exp", marker="D", alpha =0.8, color=None)

    plot_data(
        "figure.pdf",
        [s1, s2, s3, s4, s5],
        title="Demo with More Random Data",
        xlabel="x",
        ylabel="y",
        min_contrast=1.6,
        bg_hex="#5e6a7d",
        legend_position="best",
        width=16, height=10,
        xunit='mm', yunit='mV',
        export_png=True,
    )



#plot_color_seeds((0,100), 10)
