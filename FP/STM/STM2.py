from calendar import c
from itertools import combinations
import math
from pathlib import Path
import csv
import matplotlib.pyplot as plt

import numpy as np


DATA_FILE = Path(__file__).with_name("distances.csv")


def read_rows(path: Path):
	with path.open("r", encoding="utf-8") as f:
		reader = csv.DictReader(f, delimiter="\t")
		for row in reader:
			yield {
				"dx": float(row["Δx [pm]"] ),
				"dy": float(row["Δy [pm]"] ),
				"phi_given": float(row["φ [deg]"] ),
				"R_given": float(row["R [pm]"] ),
				"dz": float(row["Δz [pm]"] ),
			}


print("dx [pm]\tdy [pm]\tphi [°]\tR [pm]\tdz [pm]")
rows = list(read_rows(DATA_FILE))

# Calculate phi and prepare data for sorting
for row in rows:
    dx = row["dx"]
    dy = row["dy"]
    phi_rad = -math.atan2(dy, dx)
    phi_deg = math.degrees(phi_rad)
    R = math.hypot(dx, dy)
    dz = row["dz"]
    
    row["phi_deg"] = phi_deg % 180
    row["R"] = R
    row["dz"] = dz

# Sort rows by phi angle
sorted_rows = sorted(rows, key=lambda x: x["phi_deg"])


grouped_rows = []
for i in range(0, len(sorted_rows), 4):
    group = sorted_rows[i:i+4]
    grouped_rows.append(group)


angles = []

for group in grouped_rows:
    phi = sum(row["phi_deg"] for row in group) / len(group)
    print(phi)
    phi_err = math.sqrt(sum((row["phi_deg"] - phi) ** 2 for row in group) / len(group))
    angles.append((phi, phi_err))
    
pairwise_diffs = []
for (i, (phi_i, err_i)), (j, (phi_j, err_j)) in combinations(enumerate(angles), 2):
    diff = (phi_i - phi_j) % 180
    diff = min(diff, 180 - diff)  # Ensure the difference is within [0, 90]
    diff_err = np.sqrt(err_i**2 + err_j**2)
    pairwise_diffs.append((i, j, diff, diff_err))

print("\nPairwise angle differences:")
for i, j, diff, diff_err in pairwise_diffs:
    print(f"{i}-{j}: {diff:6.2f} ± {diff_err:5.2f} °")
    
    
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

# angles in radians for matplotlib
phi_deg = np.array([phi for phi, err in angles], float)
theta = np.deg2rad(phi_deg)

# draw one line per angle (from r=0 to r=1)
colors = ['tab:blue', 'tab:orange', 'tab:green']
for i, (t, (phi, err)) in enumerate(zip(theta, angles)):
    color = colors[i % len(colors)]

    # main ray
    ax.plot([t, t], [0, 1], lw=2, color=color, label=f"Angle {i}")

    # mirrored ray at +π (same physical state)
    ax.plot([t + np.pi, t + np.pi], [0, 1], lw=2, color=color, alpha=0.5)
    
    ax.text(
        t, 1.05,
        f"{phi:.1f}° ± {err:.1f}°",
        ha="center",
        va="center",
        fontsize=9,
        rotation=0 if 80 <= np.rad2deg(t) <= 100 else (np.rad2deg(t) if np.rad2deg(t) <= 90 else np.rad2deg(t) - 180),
        rotation_mode="anchor",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7)
    )

    # angular uncertainty wedge around the ray (fill_betweenx is correct here)
    err_rad = np.deg2rad(err)
    ax.fill_betweenx(
        [0, 1],               # r from 0 to 1
        t - err_rad,          # theta - Δ
        t + err_rad,          # theta + Δ
        color=color, alpha=0.3
    )


base_r = 0.3

for k, (i, j, diff, diff_err) in enumerate(pairwise_diffs):
    t1 = theta[i]
    t2 = theta[j]
    d = abs(t2 - t1) if abs(t2 - t1) <= np.pi/2 else abs(t2 - t1) - np.pi
    t_start = t1
    t_end   = t1 + d

    arc_t = np.linspace(t_start, t_end, 100)
    r_arc = np.full_like(arc_t, base_r + 0.05 * k)  # stagger arcs to avoid overlap

    ax.plot(arc_t, r_arc, linestyle="-", linewidth=1, color="tab:purple")
    ax.fill_between(arc_t, 0, r_arc, alpha=0.5, color="tab:purple")

    # label at mid-angle of the arc
    t_mid = (t_start + t_end) / 2
    r_mid = base_r + 0.05 * k
    ax.text(t_mid, r_mid + 0.02, f"{diff:.1f}°", ha="center", va="bottom", fontsize=8, bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7)
)

# remove radial labels and ticks
ax.set_yticklabels([])
ax.set_title("", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(Path(__file__).with_name("STM_angles.pdf"), dpi=300, transparent=True)
plt.close()

