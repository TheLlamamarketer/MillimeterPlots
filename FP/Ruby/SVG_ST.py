import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import xml.etree.ElementTree as ET


SVG_NS = "{http://www.w3.org/2000/svg}"
svg_path = Path("FP/Ruby/Tanabe_Sugano_Diagram_d3.svg")


def get_stroke(p):
    stroke = p.get("stroke")
    if not stroke:
        style = p.get("style", "")
        m = re.search(r"stroke:([^;]+)", style)
        if m:
            stroke = m.group(1).strip()
    return stroke

def is_curve_path(p):
    d = p.get("d", "")
    stroke = get_stroke(p)
    d_stripped = d.lstrip()
    return (
        d
        and len(d) > 200
        and stroke not in (None, "none")
        and d_stripped.startswith("m ")  # <- only relative data curves
    )

def iter_curve_paths(root):
    """Yield SVG <path> elements that look like the actual curves."""
    for p in root.findall(f".//{SVG_NS}path"):
        if is_curve_path(p):
            yield p


_cmd_re = re.compile(r"[a-zA-Z]|[-+]?\d*\.\d+|[-+]?\d+")


def parse_gnuplot_path(d):
    """
    Parse gnuplot-style 'm x,y dx,dy dx,dy ...' (+ optional 'h dx ...')
    into an array of absolute (x, y) points.
    """
    tokens = _cmd_re.findall(d)
    pts = []
    i = 0
    cmd = None
    x = y = None

    while i < len(tokens):
        t = tokens[i]
        if t.isalpha():
            cmd = t
            i += 1
            continue

        if cmd == "m":
            if x is None:
                x = float(t)
                y = float(tokens[i + 1])
                i += 2
                pts.append((x, y))
            else:
                dx = float(t)
                dy = float(tokens[i + 1])
                i += 2
                x += dx
                y += dy
                pts.append((x, y))

        elif cmd == "h":
            dx = float(t)
            i += 1
            if x is None or y is None:
                raise ValueError("Got 'h' before initial 'm'")
            x += dx
            pts.append((x, y))

        else:
            raise ValueError(f"Unhandled SVG path command: {cmd}")

    return np.asarray(pts)


def make_affine_mapping():
    """
    Return (ax, bx, ay, by) for mapping SVG coords -> data coords.

    x_data = ax * x_svg + bx
    y_data = ay * y_svg + by
    """
    # measured from the SVG (Inkscape etc.)
    x_svg0, x_data0 = 64.14, 0.0
    x_svg1, x_data1 = 356.22, 4.0

    y_svg0, y_data0 = 506.4, 0.0
    y_svg1, y_data1 = 18.010, 80.0

    ax = (x_data1 - x_data0) / (x_svg1 - x_svg0)
    bx = x_data0 - ax * x_svg0

    ay = (y_data1 - y_data0) / (y_svg1 - y_svg0)
    by = y_data0 - ay * y_svg0

    return ax, bx, ay, by


def svg_to_data(pts_svg, ax, bx, ay, by):
    x_svg = pts_svg[:, 0]
    y_svg = pts_svg[:, 1]
    x = ax * x_svg + bx
    y = ay * y_svg + by
    return np.column_stack((x, y))




# --- load and parse SVG ---
tree = ET.parse(svg_path)
root = tree.getroot()

curves_svg = {}
for p in iter_curve_paths(root):
    cid = p.get("id") or p.get("stroke")
    curves_svg[cid] = parse_gnuplot_path(p.get("d"))

# --- map to data coordinates ---
ax, bx, ay, by = make_affine_mapping()
curves_data = {k: svg_to_data(v, ax, bx, ay, by) for k, v in curves_svg.items()}

# --- build splines for the three states we care about ---
splines = {}
for label, pts in curves_data.items():
    x, y = pts[:, 0], pts[:, 1]
    order = np.argsort(x)
    x, y = x[order], y[order]
    splines[label] = CubicSpline(x, y)

spline_A2 = splines["path1041"]   # 4A2
spline_T2 = splines["path1034"]  # 4T2
spline_T1 = splines["path1048"]  # 4T1



'''
print("Found curves:", ", ".join(curves_svg.keys()))
x_new = np.linspace(0, 4, 500)

plt.figure()
for spline in splines.values():
    plt.plot(x_new, spline(x_new), alpha=0.5)

plt.figure()
for spline, name in [(s0, r"$^4A_2$"), (sT2, r"$^4T_2$"), (sT1, r"$^4T_1$")]:
    plt.plot(x_new, spline(x_new), label=name)

plt.xlabel("Dq/B")
plt.ylabel("E/B")
plt.title("Tanabe-Sugano Diagram for d³ Configuration")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
'''