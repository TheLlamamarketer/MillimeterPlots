# tables.py
from dataclasses import is_dataclass
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import math
from Functions.help import round_val  

# ----------------- small helpers -----------------
def _is_seq(x) -> bool:
    return isinstance(x, (list, tuple, np.ndarray))

def _len_or_one(x) -> int:
    if _is_seq(x):
        return len(x)
    return 1 if (x is not None and x != "") else 0

def _at(x, i, default=None):
    if _is_seq(x):
        return x[i] if i < len(x) else default
    return x if i == 0 else default

def _normalize_rowwise(data: dict, headers: dict) -> dict:
    """If data is {'A': {'x':..,'y':..}, 'B': {...}}, turn it into column arrays.
       Also auto-build 'name' from the outer keys if requested.
    """
    if not (isinstance(data, dict) and data and all(isinstance(v, dict) for v in data.values())):
        return data  # already column-wise or something else

    row_keys = list(data.keys())             # ["A","B","C",...]
    # list of header keys in order we will need
    needed_cols = list(headers.keys())

    col_data = {}
    for hk in needed_cols:
        if hk == "name" and all(hk not in v for v in data.values()):
            col_data["name"] = row_keys
            continue
        # pull from inner dicts (default to empty string if missing)
        col_data[hk] = [data[r].get(hk, "") for r in row_keys]
    return col_data

def _sci_parts(x: float) -> tuple[float, int]:
    """Return (mantissa, exponent) so that x = mantissa * 10**exponent."""
    if x == 0 or not np.isfinite(x):
        return (0.0, 0)
    k = int(math.floor(math.log10(abs(x))))
    return (x / (10 ** k), k)

def _fmt_num(x: float, p: int | None, *, hi: float = 1e4, lo: float = 1e-3) -> str:
    """Fixed decimals when lo <= |x| < hi, otherwise scientific; no + sign in exponent."""
    if x == 0 or not np.isfinite(x):
        return "0"
    ax = abs(x)
    if lo <= ax < hi:
        if p is None:
            return f"{x:.3g}"
        if p <= 0:
            return f"{x:.0f}"
        return f"{x:.{p}f}"
    # scientific fallback for lone numbers (no uncertainty)
    m, k = _sci_parts(x)
    # pick a sensible precision
    dec = 3 if p is None else max(0, p + k)  # tie to decimal resolution when possible
    s = f"{m:.{dec}f}" if dec > 0 else f"{m:.0f}"
    # no plus sign in exponent:
    return f"{s}e{k}"

def _fmt_pair_with_shared_exp(v: float, e: float, p: int | None, *, hi: float = 1e4, lo: float = 1e-3) -> str:
    """
    Format value ± error:
    - If magnitudes are normal, use fixed (like your old code) honoring p.
    - Otherwise choose exponent from the ERROR and print (mant ± mant_err)eK
      with mantissa precision linked to your round_val 'power'.
    """
    av = abs(v); ae = abs(e)
    if e == 0 or not np.isfinite(e):
        return _fmt_num(v, p, hi=hi, lo=lo)  # no error -> single number

    # normal-sized: fixed formatting
    if (lo <= max(av, ae) < hi):
        if p is None:
            return f"{v:.3g} \\pm {e:.3g}"
        if p <= 0:
            return f"{v:.0f} \\pm {e:.0f}"
        return f"{v:.{p}f} \\pm {e:.{p}f}"

    # scientific block: choose exponent from the ERROR
    # so we get e.g. (100 ± 8)e8 for v=1e10, e=8e8
    _, k = _sci_parts(e if e != 0 else v)
    scale = 10 ** k
    mv, me = v / scale, e / scale

    # decimals for mantissas: tie to 'p' (your round_val decimal place) shifted by k
    # This makes integers when the error rounds to 1e8, etc.
    dec = 3 if p is None else max(0, p + k)
    sv = f"{mv:.{dec}f}" if dec > 0 else f"{mv:.0f}"
    se = f"{me:.{dec}f}" if dec > 0 else f"{me:.0f}"
    return f"({sv} \\pm {se})e{k}"






# ----------------- core: STANDARD -----------------
def print_standard_table(
    data: Dict[str, Any],
    headers: Dict[str, Dict[str, Any]],
    header_groups: List[Tuple[str, int]] | None = None,
    caption: str | None = None,
    label: str | None = None,
    column_formats: List[str] | None = None,
    si_setup: str | None = None,
    show: bool = True,
) -> None:
    """
      - Cells are in math mode: {$ 1.23 \\pm 0.04 $}.
      - Respects 'label', 'err', 'data', 'dark', 'intermed', 'repeat', 'round'.
      - 'column_formats' uses S[table-format=...] unless fmt == 'l' (text col).
    """
    if not show:
        return

    data = _normalize_rowwise(data, headers)

    # column layout
    keys = list(headers.keys())
    parts = []
    if column_formats:
        for key, fmt in zip(keys, column_formats):
            dark = headers[key].get("dark", False)
            f = fmt.strip()
            if f.startswith("S["):
                col = f
            elif f in ("l", "c", "r"):      
                col = f
            else:                        
                col = f"S[table-format={f}]"
            if dark:
                col = f">{{\\columncolor{{black!20}}}}{col}"
            parts.append(col)
    else:
        for key in keys:
            dark = headers[key].get("dark", False)
            col = f"S[{si_setup}]" if si_setup else "c"
            if dark:
                col = f">{{\\columncolor{{black!20}}}}{col}"
            parts.append(col)

    layout = "| " + " | ".join(parts) + " |"

    # preamble
    print(r"\begin{table}[ht!]")
    print(r"    \centering")
    if caption:
        print(f"    \\caption{{{caption}}}")
    if si_setup and not column_formats:
        print(f"    \\sisetup{{{si_setup}}}")
    print(f"    \\begin{{tabular}}{{{layout}}}")
    print(r"    \toprule")

    # grouped header row (optional)
    if header_groups:
        group_row = " & ".join([f"\\multicolumn{{{span}}}{{|c|}}{{{name}}}" for name, span in header_groups])
        print(f"    {group_row} \\\\")
        print(r"    \midrule")

    # header labels
    head = " & ".join(entry["label"] for entry in headers.values()) + " \\\\"
    print(f"    {head}")
    print(r"    \midrule")

    # max rows
    max_rows = 0
    for k, entry in headers.items():
        col = entry.get("data", data.get(k, []))
        max_rows = max(max_rows, _len_or_one(col))

    # data rows (no per-row midrule)
    for i in range(max_rows):
        row_cells = []
        for k, entry in headers.items():
            try:
                col = entry.get("data", data.get(k, []))
                repeat = entry.get("repeat", True)
                intermed = entry.get("intermed", False)
                do_round = entry.get("round", True)

                # value
                if _is_seq(col):
                    val = _at(col, i, None)
                else:
                    val = col if (repeat or i == 0) else None

                if val is None or val == "":
                    cell = ""
                elif isinstance(val, str):
                    # leave as plain text in braces so it renders in S or c
                    cell = "{" + val + "}"
                else:
                    # error extraction (entry-local, like your original)
                    err = entry.get("err", 0)
                    if isinstance(err, (list, np.ndarray)):
                        err_val = _at(err, i, 0)
                        if isinstance(err_val, (list, np.ndarray)):
                            err_val = err_val[0] if len(err_val) > 0 else 0
                    elif isinstance(err, dict):
                        err_val = err.get(i, 0)
                    elif isinstance(err, (int, float)):
                        err_val = err
                    else:
                        err_val = 0.0

                    if not do_round:
                        cell = "{$" + _fmt_pair_with_shared_exp(float(val), float(err_val), None) + "$}"
                    else:
                        if err_val != 0:
                            v, e, p = round_val(val, err=err_val, intermed=intermed)
                            cell = "{$" + _fmt_pair_with_shared_exp(v, e, p) + "$}"
                        else:
                            v, _, p = round_val(val, err=0, intermed=intermed)
                            cell = "{$" + _fmt_num(v, p) + "$}"

                row_cells.append(cell)
            except Exception as e:
                print(f"% Error processing row {i}, column {k}: {e}")
                row_cells.append("")
        print("    " + " & ".join(row_cells) + r" \\")

    # closing
    print(r"    \bottomrule")
    print(r"    \end{tabular}")
    if label:
        print(f"    \\label{{{label}}}")
    print(r"\end{table}")
    return







# ----------------- core: COMPLEX (multi-block) -----------------
def print_complex_table(
    data: Dict[str, Any] | List[Dict[str, Any]],
    headers: Dict[str, Dict[str, Any]],
    header_groups: List[Tuple[str, int]] | None = None,
    caption: str | None = None,
    label: str | None = None,
    column_formats: List[str] | None = None,
    si_setup: str | None = None,
    show: bool = True,
) -> None:
    """
    Multi-block variant. Exactly your old styling:
      - math-mode cells with \\pm.
      - Per-block 'err' dict can override per-entry 'err'.
    """
    if not show:
        return

    # layout
    keys = list(headers.keys())
    parts = []
    if column_formats:
        for key, fmt in zip(keys, column_formats):
            dark = headers[key].get("dark", False)
            f = fmt.strip()
            if f.startswith("S["):          
                col = f
            elif f in ("l", "c", "r"):      
                col = f
            else:                        
                col = f"S[table-format={f}]"
            if dark:
                col = f">{{\\columncolor{{black!20}}}}{col}"
            parts.append(col)
    else:
        for key in keys:
            dark = headers[key].get("dark", False)
            col = f"S[{si_setup}]" if si_setup else "c"
            if dark:
                col = f">{{\\columncolor{{black!20}}}}{col}"
            parts.append(col)
    layout = "| " + " | ".join(parts) + " |"

    # preamble
    print(r"\begin{table}[ht!]")
    print(r"    \centering")
    if caption:
        print(f"    \\caption{{{caption}}}")
    if si_setup and not column_formats:
        print(f"    \\sisetup{{{si_setup}}}")
    print(f"    \\begin{{tabular}}{{{layout}}}")
    print(r"    \toprule")

    # groups + header row
    if header_groups:
        group_row = " & ".join([f"\\multicolumn{{{span}}}{{|c|}}{{{name}}}" for name, span in header_groups])
        print(f"    {group_row} \\\\")
        print(r"    \midrule")
    head = " & ".join(headers[k]["label"] for k in keys) + " \\\\"
    print(f"    {head}")
    print(r"    \midrule")

    blocks = data if isinstance(data, list) else [data]

    for bi, block in enumerate(blocks):
        block_err = block.get("err", {})

        # max rows for this block
        max_rows = 0
        for k in keys:
            col = headers[k].get("data", block.get(k, []))
            max_rows = max(max_rows, _len_or_one(col))

        # rows (no midrule inside a block)
        for i in range(max_rows):
            row_cells = []
            for k in keys:
                try:
                    entry = headers[k]
                    col = entry.get("data", block.get(k, []))
                    repeat = entry.get("repeat", True)
                    intermed = entry.get("intermed", False)
                    do_round = entry.get("round", True)

                    if _is_seq(col):
                        val = _at(col, i, "")
                    else:
                        val = col if (repeat or i == 0) else ""

                    if isinstance(val, str):
                        cell = "{" + val + "}"
                    elif val == "":
                        cell = ""
                    else:
                        # error priority: block overrides entry
                        if isinstance(block_err, dict) and k in block_err:
                            err = block_err[k]
                        else:
                            err = entry.get("err", 0)

                        if isinstance(err, (list, np.ndarray)):
                            err_val = _at(err, i, 0)
                            if isinstance(err_val, (list, np.ndarray)):
                                err_val = err_val[0] if len(err_val) > 0 else 0
                        elif isinstance(err, dict):
                            err_val = err.get(i, 0)
                        elif isinstance(err, (int, float)):
                            err_val = err
                        else:
                            err_val = 0

                        if not do_round:
                            cell = "{$" + _fmt_pair_with_shared_exp(float(val), float(err_val), None) + "$}"
                        else:
                            if err_val != 0:
                                v, e, p = round_val(val, err=err_val, intermed=intermed)
                                cell = "{$" + _fmt_pair_with_shared_exp(v, e, p) + "$}"
                            else:
                                v, _, p = round_val(val, err=0, intermed=intermed)
                                cell = "{$" + _fmt_num(v, p) + "$}"
                                
                    row_cells.append(cell)
                except Exception as e:
                    print(f"% Error processing row {i}, column {k}: {e}")
                    row_cells.append("")
            print("    " + " & ".join(row_cells) + r" \\")
        # single midrule BETWEEN blocks only
        if bi < len(blocks) - 1:
            print(r"    \midrule")

    print(r"    \bottomrule")
    print(r"    \end{tabular}")
    if label:
        print(f"    \\label{{{label}}}")
    print(r"\end{table}")
    return





# --------------- optional: DatasetSpec → blocks ---------------
def datasets_to_table_blocks(
    datasets: Sequence[Any],
    *,
    include_x: bool = True,
    include_y: bool = True,
    include_err: bool = True,
) -> List[Dict[str, Any]]:
    """Accepts your DatasetSpec (or dicts with x,y,yerr,label) and returns blocks for print_complex_table."""
    blocks: List[Dict[str, Any]] = []
    for ds in datasets:
        if is_dataclass(ds):
            x = getattr(ds, "x", None)
            y = getattr(ds, "y", None)
            yerr = getattr(ds, "yerr", None)
            label = getattr(ds, "label", None)
        else:
            x = ds.get("x"); y = ds.get("y")
            yerr = ds.get("yerr", None); label = ds.get("label", None)

        b: Dict[str, Any] = {}
        if include_x and x is not None:
            b["x"] = np.asarray(x) if _is_seq(x) else [x]
        if include_y and y is not None:
            b["y"] = np.asarray(y) if _is_seq(y) else [y]
        if include_err and yerr is not None:
            b["err"] = {"y": yerr}
        if label:
            b["name"] = label
        blocks.append(b)
    return blocks
