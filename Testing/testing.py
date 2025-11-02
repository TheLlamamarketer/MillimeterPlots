import numpy as np
from Functions.tables import datasets_to_table_blocks, print_standard_table, print_complex_table

data = {
    "name": ["Al", "Cu", "Si", "Ge"],          # 4 rows, text column
    "n":    [2.713e19, 8.49e22, 2.0e10, 2.3e13],  # numeric
    "mu":   1500.0,                              # scalar column
}

headers = {
    "name": {"label": "{Material}", "dark": True, "data": ["Al", "Cu", "Si", "Ge"]},
    "n":    {"label": "{$n$ ($\\mathrm{cm^{-3}}$)}", "err": [1e18, 5e21, 1e9, 1e12], "intermed": True},
    "mu":   {"label": "{$\\mu$ ($\\mathrm{cm^2/Vs}$)}", "err": 50.0, "repeat": False},   # shown only in first row
}

# Column formats: text + numeric shorthand + explicit S
column_formats = ["l", "2.2", "S[table-format=4.0]"]

print_standard_table(
    data=data,
    headers=headers,
    header_groups=[("ID", 1), ("Transport", 2)],     # grouped header row
    caption="Carrier density $n$ and mobility $\\mu$ for selected materials.",
    label="tab:transport",
    column_formats=column_formats,
    si_setup=None,   # ignored since column_formats is provided
    show=True,
)


block1 = {
    "T": np.array([300.0, 320.0, 340.0]),
    "R": np.array([10.01,  9.56,  9.12]),            # resistance
    "err": {
        "T": 0.2,                                     # per-block scalar error for all T rows
        "R": np.array([0.05, 0.05, 0.04]),            # per-block vector error for R
    }
}
block2 = {
    "T": np.array([360.0, 380.0]),
    "R": np.array([8.79,  8.51]),
    "bias": 1.000,                                    # scalar: repeat will control display
    "err": {
        "T": [0.2, 0.2],                              # per-block list error
        # no R error here → will fall back to header-level if provided, else 0
    }
}

headers = {
    "T":    {"label": "{$T$ (K)}", "intermed": False},
    "R":    {"label": "{$R$ (\\si{\\ohm})}", "intermed": True, "err": 0.02},  # header-level fallback err
    "bias": {"label": "{Bias}", "repeat": False, "round": False, "dark": True},  # shown once per block (top row of block)
}

# Numeric shorthand for T & R; text for Bias
column_formats = ["2.0", "2.2", "l"]

print_complex_table(
    data=[block1, block2],
    headers=headers,
    header_groups=[("Run A", 2), ("Config", 1)],
    caption="Two runs with per-block uncertainties (block errors override header errors).",
    label="tab:runs",
    column_formats=column_formats,
    si_setup=None,
    show=True,
)