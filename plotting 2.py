"""
Plotting utilities focused on crisp aesthetics, reproducibility, and flexible input formats.
Key features:
- Deterministic, background-aware color palettes with minimum contrast enforcement.
- Group-aware coloring with stable variations per series (no reliance on Python object id()).
- Clean plot styling (light/dark), better legends, optional duplicate-point aggregation.
- Friendly inputs: accepts bare (x, y), dicts, or DatasetSpec dataclass.
- Safer errorbar handling (sym/asym), confidence bands, and fit overlays.
- Returns (fig, ax) for further customization; optional multi-format export (PDF/PNG/SVG).
"""

from dataclasses import dataclass, field
from random import seed
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import hashlib
import itertools
import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.ticker import AutoMinorLocator
import matplotlib.colors as mcolors
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
    marker: str = "."                      # "o", "s", "D", "X", "P", ".", "v", "^", "<", ">", "1", "2", "3", "4", "8"
    line: str = "None"                     # "-", "--", ":", "None", "steps"
    linewidth: float = 1.4
    markersize: float = 24
    alpha: float = 0.95
    zorder: int = 3

    # error bars (sym or asym). Each can be scalar, array, or (lower, upper)
    yerr: float | Sequence[float] | Tuple[Sequence[float], Sequence[float]] | Tuple[float, float] | None = None
    xerr: float | Sequence[float] | Tuple[Sequence[float], Sequence[float]] | Tuple[float, float] | None = None

    # fits & intervals
    fit_y: Sequence[float] | None = None         # same length as x or fit_x
    fit_x: Sequence[float] | None = None
    fit_line: str = "-"
    fit_label: bool = True
    fit_error_lines: List[Tuple[Sequence[float] | None, Sequence[float] | None]] | None = None

    # confidence intervals: list of (lower, upper) arrays matching x (or fit_x)
    confidence: List[Tuple[Sequence[float] | None, Sequence[float] | None]] | None = None
    confidence_label: bool = True

    # misc
    aggregate_duplicates: bool = False      # turn duplicated (x,y) samples into larger markers


# -----------------------------------------------------------------------------
# Color helpers
# -----------------------------------------------------------------------------

_DEF_OKABE_ITO = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9",
    "#CC79A7", "#F0E442", "#000000"
]


def _stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) & 0x7FFFFFFF


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    # c in [0,1]
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


def _ensure_contrast(rgb: Tuple[float, float, float], bg_rgb: Tuple[float, float, float],
                     min_ratio: float = 3.0) -> Tuple[float, float, float]:
    """Adjust V in HSV to meet a minimum WCAG-like contrast vs. background."""
    cr = _contrast_ratio(rgb, bg_rgb)
    if cr >= min_ratio:
        return rgb
    # If bg is light, darken; if dark, lighten
    bg_L = _relative_luminance(bg_rgb)
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    for step in np.linspace(0.02, 0.7, 35):
        v_try = np.clip(v - step, 0, 1) if bg_L > 0.5 else np.clip(v + step, 0, 1)
        cand = colorsys.hsv_to_rgb(h, s, v_try)
        if _contrast_ratio(cand, bg_rgb) >= min_ratio:
            return cand
    print("No suitable contrast found, returning original color.")
    return rgb  # best effort


def generate_palette(n: int,
                     bg_hex: str = "#FFFFFF",
                     seed: int | None = 203,
                     saturation: float = 0.72,
                     value: float = 0.82,
                     method: str = "golden",
                     variation: float = 0.07,
                     min_contrast: float = 3.0,
                     base_palette: list[str] | None = None) -> list[str]:
    """Create n distinct colors that contrast with background.

    method: "golden" (hue steps by golden angle), "linspace" (even spread), or "okabe" for colorblind-friendly palettes.
    base_palette: if provided and length>=n, truncates and returns that.
    """
    bg_rgb = mcolors.to_rgb(bg_hex)

    if base_palette and len(base_palette) >= n:
        cols = base_palette[:n]
        return [mcolors.to_hex(_ensure_contrast(mcolors.to_rgb(c), bg_rgb, min_contrast)) for c in cols]

    if method == "okabe":
        cols = _DEF_OKABE_ITO * ((n + len(_DEF_OKABE_ITO) - 1) // len(_DEF_OKABE_ITO))
        return [mcolors.to_hex(_ensure_contrast(mcolors.to_rgb(c), bg_rgb, min_contrast)) for c in cols[:n]]

    rng = np.random.default_rng(seed)
    base = rng.random()

    if method == "golden":
        step = (math.sqrt(5) - 1) / 2.0  # ~0.618...
        hues = (base + step * np.arange(n)) % 1.0
    else:  # linspace
        hues = (base + np.linspace(0, 1, n, endpoint=False)) % 1.0

    # jitter hues a bit for natural variety
    hues = (hues + rng.uniform(-variation, variation, size=n)) % 1.0

    rgb_list = [colorsys.hsv_to_rgb(h, saturation, value) for h in hues]
    rgb_list = [_ensure_contrast(c, bg_rgb, min_ratio=min_contrast) for c in rgb_list]
    return [mcolors.to_hex(c) for c in rgb_list]


def colors_from_groups(datasets: Sequence[DatasetSpec | Mapping[str, Any]],
                       color_seed: int | None = 203,
                       bg_hex: str = "#FFFFFF",
                       hue_variation_within_group: float = 0.04,
                       palette_method: str = "golden",
                       min_contrast: float = 3.0) -> Dict[str, str]:
    """Assign a color to each dataset keyed by its stable dataset key.

    Returns a dict: key -> hex color.
    """
    # Normalize into DatasetSpec views to read keys/groups/colors
    norm: List[DatasetSpec] = [normalize_dataset(d) for d in datasets]

    # Build groups
    groups: Dict[str, List[DatasetSpec]] = {}
    singles: List[DatasetSpec] = []

    for d in norm:
        if d.color is not None:
            continue  # already explicitly colored
        if d.color_group:
            groups.setdefault(d.color_group, []).append(d)
        else:
            singles.append(d)

    # Prepare a palette for distinct groups + singles
    n_groups = len(groups)
    n_singles = len(singles)
    base_palette = generate_palette(n_groups + n_singles, bg_hex=bg_hex, seed=color_seed,
                                    method=palette_method, min_contrast=min_contrast)

    # Assign base colors to groups then singles
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
        for j, d in enumerate(items):
            if d.color is None:  # still allowed to override at dict level
                hj = (h + rng.uniform(-hue_variation_within_group, hue_variation_within_group)) % 1.0
                c = colorsys.hsv_to_rgb(hj, s, v)
                c = _ensure_contrast(c, mcolors.to_rgb(bg_hex), min_ratio=min_contrast)
                out[_ds_key(d)] = mcolors.to_hex(c)

    for d in singles:
        out[_ds_key(d)] = base_palette[idx]
        idx += 1

    # Respect explicit colors last
    for d in norm:
        if d.color is not None:
            out[_ds_key(d)] = mcolors.to_hex(d.color)

    return out


# -----------------------------------------------------------------------------
# Input normalization
# -----------------------------------------------------------------------------

_DEF_MARKERS = ["o", "s", "^", "D", "P", "X", "+", "*", "v", "<", ">", "h"]


def normalize_dataset(d: DatasetSpec | Mapping[str, Any] | Tuple[Sequence[float], Sequence[float]] | Tuple[Sequence[float], Sequence[float], str]) -> DatasetSpec:
    """Accept several input forms and return a DatasetSpec.
    Supported:
      - DatasetSpec(...)
      - {"x": ..., "y": ..., optional keys}
      - (x, y)
      - (x, y, label)
    """
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
        # keys compatibility with user's previous structure
        x = d.get("x") or d.get("xdata")
        y = d.get("y") or d.get("ydata")
        if x is None or y is None:
            raise ValueError("Dataset dict must contain 'x'/'xdata' and 'y'/'ydata'")
        spec = DatasetSpec(
            x=x,
            y=y,
            label =                 d.get("label"),
            color =                 d.get("color"),
            color_group =           d.get("color_group"),
            key =                   d.get("key"),
            marker =                d.get("marker", d.get("marker", ".")),
            line =                  d.get("line", d.get("line", "None")),
            linewidth =             d.get("linewidth", 1.4),
            markersize =            d.get("markersize", 24),
            alpha =                 d.get("alpha", 0.95),
            zorder =                d.get("zorder", 3),
            yerr =                  d.get("yerr"),
            xerr =                  d.get("xerr"),
            fit_y =                 d.get("fit") or d.get("fit_y"),
            fit_x =                 d.get("fit_xdata") or d.get("fit_x"),
            fit_line =              d.get("fit_line", "-"),
            fit_label =             d.get("fit_label", True),
            fit_error_lines =       d.get("fit_error_lines"),
            confidence =            d.get("confidence"),
            confidence_label =      d.get("confidence_label", True),
            aggregate_duplicates =  d.get("aggregate_duplicates", False),
        )
        return spec

    raise TypeError("Unsupported dataset type. Use DatasetSpec, dict, or (x,y).")


def _ds_key(d: DatasetSpec) -> str:
    if d.key:
        return d.key
    base = d.label or "series"
    cg = d.color_group or "nogroup"
    # Stable pseudo-key
    return f"{base}|{cg}|{_stable_int(str(len(np.atleast_1d(d.x))))}"


# -----------------------------------------------------------------------------
# Grid & style helpers
# -----------------------------------------------------------------------------

_LIGHT_RC = {
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


def _apply_bg(ax: mpl.axes.Axes, bg_hex: str):
    ax.set_facecolor(bg_hex)
    ax.figure.set_facecolor(bg_hex)


# -----------------------------------------------------------------------------
# Main plotting
# -----------------------------------------------------------------------------

def plot_data(
    filename: str | None,
    datasets: Sequence[DatasetSpec | Mapping[str, Any] | Tuple[Any, Any] | Tuple[Any, Any, str]],
    *,
    color_seed: int | None = 203,
    title: str | None = None,
    subtitle: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend_position: str | None = "best",
    width_cm: float = 16,
    height_cm: float = 10,
    dpi: int = 200,
    style: str = "light",  # "light" | "dark"
    bg_hex: str = "#FFFFFF",
    palette_method: str = "golden",
    min_contrast: float = 3.0,
    xticks: Tuple[float, float] | None = None,   # (step, offset)
    yticks: Tuple[float, float] | None = None,   # (step, offset)
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    tight: bool = True,
    transparent: bool = False,
    export_svg: bool = False,
    export_png: bool = False,
    legend_ncol: int | None = None,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Plot multiple datasets with clean defaults. Save to filename if given.

    Returns (fig, ax).
    """
    # Normalize datasets and filter out empties early
    specs = [normalize_dataset(d) for d in datasets]
    specs = [s for s in specs if s.x is not None and s.y is not None]

    # Resolve colors
    color_map = colors_from_groups(
        specs, color_seed=color_seed, bg_hex=bg_hex,
        palette_method=palette_method, min_contrast=min_contrast
    )

    # Styling context
    rc = {**(_LIGHT_RC if style == "light" else _DARK_RC)}
    with mpl.rc_context(rc):
        fig, ax = plt.subplots(figsize=(width_cm/2.54, height_cm/2.54), dpi=dpi, constrained_layout=True)
        _apply_bg(ax, bg_hex)

        if xticks is not None and yticks is not None and xlim is not None and ylim is not None:
            _draw_dense_grid(ax, xlim, ylim, width_cm, height_cm, bg_hex, xticks, yticks)
        else:
            ax.grid(True, which="major")
            ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.8)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        # Plot series
        for i, s in enumerate(specs):
            x = np.atleast_1d(np.asarray(s.x, dtype=float))
            y = np.atleast_1d(np.asarray(s.y, dtype=float))
            if x.size == 0 or y.size == 0:
                continue
            if x.shape != y.shape:
                # allow (1, N) vs (N,) mismatches
                y = y.reshape(x.shape)

            color = s.color or color_map[_ds_key(s)]

            # Confidence bands (behind everything)
            if s.confidence is not None:
                for k, (lo, hi) in enumerate(s.confidence):
                    if lo is None or hi is None:
                        continue
                    xci = np.asarray(s.fit_x if s.fit_x is not None else x, dtype=float)
                    lo = np.asarray(lo, dtype=float)
                    hi = np.asarray(hi, dtype=float)
                    alpha = max(0.1, 0.35 - 0.08 * k)
                    ax.fill_between(xci, lo, hi, facecolor=color, edgecolor=None, 
                                    alpha=alpha, zorder=2, label=s.label + f"{k+1}σ Confidence Interval" if s.confidence_label and s.label else None)

            # Fit lines
            if s.fit_y is not None:
                xf = np.asarray(s.fit_x if s.fit_x is not None else x, dtype=float)
                yf = np.asarray(s.fit_y, dtype=float)
                label_fit = f"{s.label} Fit" if (s.label and s.fit_label) else None
                ax.plot(xf, yf, linestyle=s.fit_line, color=color, linewidth=max(1.0, s.linewidth),
                        label=label_fit, zorder=s.zorder)

            # Fit error lines
            if s.fit_error_lines is not None:
                xf = np.asarray(s.fit_x if s.fit_x is not None else x, dtype=float)
                for (lo, hi) in s.fit_error_lines:
                    if lo is not None:
                        ax.plot(xf, np.asarray(lo, dtype=float), linestyle="--", color=color, linewidth=1.0, zorder=s.zorder)
                    if hi is not None:
                        ax.plot(xf, np.asarray(hi, dtype=float), linestyle="--", color=color, linewidth=1.0, zorder=s.zorder)

            # Aggregate duplicates (optional)
            if s.aggregate_duplicates:
                # counts per (x,y)
                pts, counts = np.unique(np.c_[x, y], axis=0, return_counts=True)
                x_agg, y_agg = pts[:, 0], pts[:, 1]
                sizes = 16 + 84 * (np.log1p(counts) / np.log1p(counts).max())
                if s.marker != "None":
                    ax.scatter(x_agg, y_agg, s=sizes, color=color, marker=s.marker, alpha=s.alpha, zorder=s.zorder,
                               label=s.label)
                if s.line != "None":
                    ax.plot(x_agg, y_agg, linestyle=s.line, color=color, linewidth=s.linewidth, zorder=s.zorder)
            else:
                # Standard errorbars/markers/lines
                xerr_lower, xerr_upper = _normalize_err(s.xerr, x)
                yerr_lower, yerr_upper = _normalize_err(s.yerr, y)

                if s.xerr is not None or s.yerr is not None:
                    ax.errorbar(
                        x, y,
                        xerr=None if s.xerr is None else [np.abs(xerr_lower), np.abs(xerr_upper)],
                        yerr=None if s.yerr is None else [np.abs(yerr_lower), np.abs(yerr_upper)],
                        fmt=s.marker if s.marker != "None" else "o",
                        color=color, ecolor=color, elinewidth=0.9, capsize=3, capthick=0.9,
                        markersize=max(4, s.markersize / 3), alpha=s.alpha, zorder=s.zorder,
                        label=s.label)
                else:
                    if s.marker != "None":
                        ax.scatter(x, y, color=color, marker=s.marker, s=s.markersize, alpha=s.alpha, zorder=s.zorder,
                                   label=s.label)
                if s.line != "None":
                    ax.plot(x, y, linestyle=s.line, color=color, linewidth=s.linewidth, zorder=s.zorder,
                            label=None if s.marker != "None" else s.label)

        # Axes labels & title
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

        # Legend (deduplicate labels)
        if legend_position:
            handles, labels = ax.get_legend_handles_labels()
            dedup = {}
            for h, l in zip(handles, labels):
                if l and l not in dedup:
                    dedup[l] = h
            if dedup:
                ncol = legend_ncol or (1 if len(dedup) < 5 else 2)
                ax.legend(dedup.values(), dedup.keys(), loc=legend_position, frameon=False, ncol=ncol)

        if tight:
            fig.tight_layout()

        # Save/Export
        if filename:
            fig.savefig(filename, bbox_inches="tight", dpi=dpi, transparent=transparent)
        if export_png and filename:
            fig.savefig(_with_ext(filename, ".png"), bbox_inches="tight", dpi=max(200, dpi), transparent=transparent)
        if export_svg and filename:
            fig.savefig(_with_ext(filename, ".svg"), bbox_inches="tight", dpi=dpi, transparent=transparent)

        return fig, ax


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
    # array-like symmetric
    e = np.asarray(err, dtype=float)
    return e, e


def _draw_dense_grid(ax: mpl.axes.Axes,
                     xlim: Tuple[float, float], ylim: Tuple[float, float],
                     width_cm: float, height_cm: float,
                     bg_hex: str,
                     xticks: Tuple[float, float], yticks: Tuple[float, float]):
    """Tri-level grid similar to your original but faster via LineCollection."""
    xmin, xmax = xlim
    ymin, ymax = ylim

    # aspect
    aspect_ratio = (xmax - xmin) / max(1e-12, (ymax - ymin)) * (height_cm / max(1e-12, width_cm))
    ax.set_aspect(aspect_ratio, adjustable='box')

    xstep, xoff = xticks
    ystep, yoff = yticks

    # Build coordinates
    xs_main = np.arange(xmin + xoff, xmax + 1e-9, xstep)
    ys_main = np.arange(ymin + yoff, ymax + 1e-9, ystep)

    # Secondary/tertiary densities
    def _dense(axis_min, axis_max, px_per_unit: float):
        # choose reasonable pixel density per figure size
        # two finer levels
        span = axis_max - axis_min
        return (
            np.arange(axis_min, axis_max, span / (px_per_unit * 10)),
            np.arange(axis_min, axis_max, span / (px_per_unit * 5)),
        )

    xs_ter, xs_sec = _dense(xmin, xmax, width_cm)
    ys_ter, ys_sec = _dense(ymin, ymax, height_cm)

    lines = []
    def _add_vlines(xs):
        for x in xs:
            lines.append([(x, ymin), (x, ymax)])
    def _add_hlines(ys):
        for y in ys:
            lines.append([(xmin, y), (xmax, y)])

    # Build line collections by width tiers
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

    # Ticks/limits
    ax.set_xticks(xs_main)
    ax.set_yticks(ys_main)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Borders
    for spine in ax.spines.values():
        spine.set_edgecolor(bg_hex)
        spine.set_zorder(2)


# -----------------------------------------------------------------------------
# Seed visualization (optional)
# -----------------------------------------------------------------------------

def plot_color_seeds(seed_range: Tuple[int, int] = (100, 110), num_groups: int = 4,
                     bg_hex: str = "#FFFFFF", palette_method: str = "golden"):
    """Visualize palette changes across seeds for group-level colors."""
    seeds = range(seed_range[0], seed_range[1] + 1)
    fig, ax = plt.subplots(figsize=(max(6, len(list(seeds)) * 0.4), num_groups * 0.6), constrained_layout=True)

    for col, seed in enumerate(seeds):
        cols = generate_palette(num_groups, bg_hex=bg_hex, seed=seed, method=palette_method)
        for row in range(num_groups):
            ax.add_patch(Rectangle((col, row), 1, 1, color=cols[row]))

    ax.set_xticks(np.arange(len(list(seeds))) + 0.5)
    ax.set_xticklabels(list(seeds), rotation=45)
    ax.set_yticks(np.arange(num_groups) + 0.5)
    ax.set_yticklabels([f"Group {i+1}" for i in range(num_groups)])
    ax.set_xlim(0, len(list(seeds)))
    ax.set_ylim(0, num_groups)
    ax.invert_yaxis()
    ax.set_frame_on(False)
    plt.show()





x = np.linspace(0, 10, 200)
y = 0.2 * (x-5)**2 + 0.1 * np.random.randn(x.size)

s1 = DatasetSpec(x=x, y=y, label="Data A", color_group="run", marker="o", aggregate_duplicates=False)
s2 = DatasetSpec(x=x, y=np.cos(x) + 0.6*np.random.randn(x.size), label="Data B", color_group="go", fit_y=np.cos(x), fit_line="-")



plot_data(
    "figure.pdf",
    [s1, s2],
    title="Demo",
    xlabel="x",
    ylabel="y",
    style="light",          # or "dark"
    bg_hex="#af3bd0",
    min_contrast=1.5,
    legend_position="best",
    width_cm=16, height_cm=10,
    export_png=True
)

