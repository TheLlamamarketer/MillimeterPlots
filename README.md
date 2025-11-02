# MillimeterPlots

Welcome to the MillimeterPlots repository — a small collection of plotting, color and table utilities and helpers used to analyze and render spectroscopic and measurement data. This README gives new users a practical quickstart, points to the bootstrap installer, and documents every function/module in the `Functions/` package so you can find and use functionality quickly.

## Quick start — prepare your environment

- If you haven't already set up the project dependencies, start with the files in `bootstrap/`.
  - The `bootstrap/README.md` explains the supported Python version and how to create and activate a virtual environment and install requirements. If you use PowerShell on Windows (default for this workspace), run the steps in that file.
  - For convenience, the repository contains `bootstrap/install_requirements.ps1` and `bootstrap/install_requirements.py` to automate environment setup.

Example (PowerShell):

```powershell
# See bootstrap/README.md for the exact commands — this is a summary.
& .\.venv\Scripts\Activate.ps1
python -m pip install -r bootstrap\requirements.txt
```

Once dependencies are installed, you can run the example scripts in the `Functions/` package or import the utilities from your own scripts.

## Project layout (relevant)

- `Functions/` — main library code. Each file is documented below.
- `bootstrap/` — environment setup helpers and a second README with installation instructions (recommended starting point for new users).
- `colormaps/` — pre-computed LUTs used by the color utilities.

## How to use this README

This file documents the public functions and main helpers inside `Functions/`. For each module you'll find:

- short description of what the module provides
- list of important functions/classes with signatures and short explanations
- minimal usage examples you can copy into a script or an interactive session

If you want to contribute: add docstrings to functions in the source files (they were used to build this README) and keep examples under `Functions/` or a new `examples/` directory.

---

## Functions package overview

Below are the modules in `Functions/` with short API descriptions and small examples. Use `from Functions import <module>` or `import Functions.<module>` to access them.

### `color_builder.py`

Purpose: Create consistent, group-aware color maps for a collection of dataset descriptors. Useful when plotting many series and you want grouped colors with slight variations.

Primary function:

- `build_color_map(all_datasets, color_seed=100, bg_hex='#FFFFFF', hue_range=0.05)`
  - Input: `all_datasets` — an iterable of dict-like dataset descriptors. Each dataset may include keys like `x`, `y`, `label`, `color`, `color_group`.
  - Behavior: Datasets that share `color_group` get the same base color; datasets with explicit `color` keep it; ungrouped datasets get individual colors.
  - Output: a mapping `{id(dataset): "#hexcolor"}` where `id(dataset)` matches the object identity used in the caller.

Example:

```python
from Functions.color_builder import build_color_map

data = [
    {'x': [0,1], 'y':[1,2], 'label':'A', 'color_group':'run1'},
    {'x': [0,1], 'y':[2,3], 'label':'B', 'color_group':'run1'},
    {'x': [0,1], 'y':[0,1], 'label':'C'},
]
color_map = build_color_map(data)
```

---

### `find_interval.py`

Purpose: Helpers that attempt to find "clean" numeric axis intervals for plotting or table layout.

Functions:

- `largest_power_of_10(n)` — returns 10**exponent for the magnitude of `n` (used for normalization).
- `find_clean_intervals(lower_bound, upper_bound, total_cm, min_increment=0.0001, max_increment=None)` — brute-forces small increments to find tidy divisions inside a numeric range; prints candidates.
- `find_interval(xdata, ydata, name=' ', xerror=None, yerror=None, width=20, height=28, max_increment_height=1, max_increment_width=1)` — higher level: prints candidate intervals for width and height based on provided data and optional errors.

Usage (interactive):

```python
from Functions.find_interval import find_interval
find_interval(xdata=[0, 1, 2], ydata=[10, 11, 12], name='example')
```

Notes: These functions print results and are handy when you want to pick axis tick steps that divide the plotted range into round numbers.

---

### `help.py`

Purpose: Numeric helpers for rounding, uncertainties, simple linear regression and model-fitting wrappers.

Key functions and utilities:

- `FirstSignificant(x)`, `LastSignificant(x)`, `last_digit(num)` — helpers to determine significant digits.
- `round_val(val, err=0, intermed=True)` — rounds `val` and its error to appropriate significant digits. Returns `(rounded_val, rounded_err, power)` or `(rounded_val, err, power)` when `err==0`.
- `print_round_val(val, err=0, intermed=True)` — returns a string like `"{val} \pm {err}"` formatted for LaTeX.
- `slope(xdata, ydata, yerr=None)` — computes weighted linear regression (returns a, da, b, db, R2, variance) where `y = a + b*x`.
- `lmfit(xdata, ydata, yerr=None, model='linear', constraints=None, const_weight=1)` — wrapper around `lmfit.Model` supporting `linear`, `quadratic`, `exponential`, `exponential_decay`, `gaussian` models. Returns the `lmfit` result object.
- `calc_CI(result, xdata, sigmas=[1])` — compute fit confidence intervals using `result.eval_uncertainty`.
- `extract_params(result)` — returns a dict of param -> (value, stderr).
- `calc_R2(result)` — convenience for `result.rsquared`.
- `print_fit_summary(result)` — prints parameter values, errors, and R².
- `plot(x,y,dy,a,b,da,db, ...)` — small convenience plotting function for linear fits with uncertainty bands.

Quick example — linear slope:

```python
from Functions.help import slope
import numpy as np
x = np.array([0,1,2,3])
y = np.array([0.1, 1.1, 2.0, 3.2])
a, da, b, db, R2, var = slope(x, y)
print(f'intercept={a}, slope={b}, R^2={R2:.4f}')
```

For model fitting use `lmfit`. Example:

```python
from Functions.help import lmfit
res = lmfit(x, y, model='linear')
print(res.fit_report())
```

---

### `plotting.py`

Purpose: High-quality plotting utilities focused on readable palettes, automatic contrast, SI/engineering axis formatting, group-aware color schemes, and a compact `plot_data` API.

Main items:

- `DatasetSpec` (dataclass) — a convenient typed container for dataset properties used by `plot_data` (x, y, label, color, color_group, marker, line, errors, fit, confidence bands, etc.).
- `generate_palette(n, bg_hex='#FFFFFF', seed=203, ...)` — generate `n` distinct colors that contrast with given background.
- `colors_from_groups(datasets, ...)` — assign stable colors per dataset or per `color_group`.
- `normalize_dataset(d)` — converts tuples or mapping objects to `DatasetSpec`.
- `plot_data(datasets, filename=None, color_seed=203, title=None, xlabel=None, ylabel=None, width=20, height=20, dpi=200, bg_hex='#FFFFFF', ...) -> (fig, ax)` — main plotting function. Accepts lists of `DatasetSpec`, dicts, or (x, y) tuples. Supports error bars, fits, confidence bands, outlines for readability, SI/Eng tick formatting, and exports.
- `plot_color_seeds(...)` — visual helper to inspect palettes vs seeds.
- `test_plotting()` — a small demo that builds and exports a demo plot.

Minimal usage:

```python
from Functions.plotting import DatasetSpec, plot_data

s1 = DatasetSpec(x=[0,1,2], y=[0, 1, 2], label='A', color_group='g1')
s2 = DatasetSpec(x=[0,1,2], y=[2, 2, 1], label='B', color_group='g2')
fig, ax = plot_data([s1, s2], title='Example', xlabel='mm', ylabel='mV', plot=True)
```

Notes: `plot_data` returns `(fig, ax)` so callers can further customize the Matplotlib axes. The plotting module emphasizes stable, reproducible color choices and readable defaults.

---

### `tables.py`

Purpose: Produce LaTeX-ready tables for numeric results including uncertainties. These helpers follow a pattern of formatting numbers and printing table code to stdout.

Important functions:

- `print_standard_table(data, headers, header_groups=None, caption=None, label=None, column_formats=None, si_setup=None, show=True)` — prints a single-block LaTeX table. `data` can be column-wise or row-wise. `headers` defines column labels, rounding and error handling.
- `print_complex_table(data, headers, ...)` — prints a multi-block table layout.
- `datasets_to_table_blocks(datasets, include_x=True, include_y=True, include_err=True)` — convert plotting datasets to table-ready blocks.

Example usage prints to console; you can capture output and paste into your LaTeX document.

```python
from Functions.tables import print_standard_table
print_standard_table(data, headers, caption='Example')
```

---

### `testing.py`

Purpose: Example driver scripts showing how to call table helpers. Not a test harness per se; run these files interactively or as modules.

They contain example `data` and `headers` and call `print_standard_table` and `print_complex_table` so you can see the output format and adapt to your needs.

Usage:

```powershell
python -m Functions.testing
```

---

### `wavelength_colors.py`

Purpose: Convert spectral wavelengths and lines to RGB, build continuous spectrum colormaps, render spectrum strips, and create colormaps for Matplotlib. Uses the `colour` package and pre-computed CMFs/illuminants.

Key functions and utilities:

- `set_base_resolution(step_nm=0.1, start=380.0, end=780.0)` — rebuilds internal lookup tables at a custom wavelength resolution.
- `srgb_eotf(lin)` — sRGB electro-optical transfer function (linear → encoded sRGB).
- `wavelength_to_rgb(wl_nm, *, space='sRGB', brightness='D65', gamut_map='oklab', encode=True)` — converts a single wavelength to RGB values.
- `render_spectrum(start=380, end=780, step=0.1, space='sRGB', brightness='D65', gamut_map='oklab', H=40)` — returns an image (height H) and the wavelength grid useful for visual strips.
- `lines_to_rgb(wavelengths_nm, intensities=1.0, **kwargs)` — sum a set of spectral lines into an RGB color.
- `draw_lines_on_strip(img, grid, wls, amps, sigma_nm=0.05, ...)` — draw narrow lines onto a spectrum image.
- `build_spectrum_cmap(name='spectrum_wavelengths', start=380, end=780, step=0.01, ...) -> (cmap, norm)` — construct a Matplotlib `ListedColormap` and `Normalize` mapping wavelength→[0,1].
- `save_cmap_lut(path, ...)`, `load_cmap(path, name='spectrum_from_file')` — save/load LUTs for faster reuse.
- `show_spectrum_strip()` — demo that renders several example strips and line overlays.

Usage example (generate and show a spectrum strip):

```python
from Functions.wavelength_colors import show_spectrum_strip
show_spectrum_strip()
```

If you want a colormap registered in Matplotlib:

```python
from Functions.wavelength_colors import build_spectrum_cmap
cmap, norm = build_spectrum_cmap()
```

The module includes many lower-level helpers for gamut-mapping and Oklab-based clipping which are advanced but exposed for reproducible spectral rendering.

---

### `__init__.py` in `Functions/`

Purpose: keep package import simple and side-effect free. Import modules explicitly when you need them, for example `from Functions import plotting` or `import Functions.wavelength_colors as wc`.

## Where to get help and examples

- Start with `bootstrap/README.md` to install dependencies.
- Run examples in `Functions/` directly with `python -m Functions.testing` or `python -m Functions.wavelength_colors` (some modules include `if __name__ == '__main__':` demo runners).
- For plotting examples, try calling `Functions.plotting.test_plotting()` after setting up a display backend or enabling `plot=True` in `plot_data`.

## Contribution notes

- Preferred small PRs: add or extend docstrings in the source file next to the function.
- If you add a new helper in `Functions/`, add at least one short usage example in the file under a `if __name__ == '__main__':` demo block or update this README.
- If you change public API, update this README's module section to keep documentation in sync.

## Verification & next steps

- I scanned the `Functions/` Python modules and used top-level docstrings and function names to build this README. If you want, I can (automatically) generate a more exhaustive reference (including function parameter lists) by parsing each function signature and docstring into an API reference file.

If you'd like that, tell me whether you want the reference as:

1. A single long `FUNCTIONS_API.md` file that lists every function signature verbatim; or
2. Individual `Functions/<module>_README.md` files alongside each module; or
3. Inline docstring normalization (e.g., convert to numpydoc style) and a generated Sphinx or mkdocs site.

---
