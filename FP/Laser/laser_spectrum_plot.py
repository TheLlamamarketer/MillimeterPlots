import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

fig, ax = plt.subplots(figsize=(20 / 2.54, 15 / 2.54))

ax.set_xlim(-4, 4)
ax.set_ylim(-0.1, 1.15)

x = np.linspace(-4, 4, 4000)

def gauss(x, xk, Ak, sigma=0.2, shift=0.1):
    return Ak * np.exp(-((x - xk + shift) ** 2) / (sigma ** 2))

y = np.zeros_like(x)

shift = 0.1
for center in [-2, 2]:
    for k in [-1, 0, 1]:
        xk = center + k * 0.9
        Ak = np.exp(-((xk - center - shift) ** 2))
        y += gauss(x, xk, Ak, sigma=0.2, shift=shift)

peaks = find_peaks(y, height=0.05)[0]
ax.plot(x, y, color="black", linewidth=1.2)

# Mode numbers below x-axis
labels = list(zip(x[peaks], [str(i) for i in range(1, len(peaks) + 1)]))
for xv, s in labels:
    ax.text(xv, 0.0, s, ha="center", va="top", clip_on=False)

# Frabipero spacing Δt
ax.annotate(
    "",
    xy=(x[peaks[-2]], 0.98), xytext=(x[peaks[1]], 0.98),
    arrowprops=dict(arrowstyle="<->", linewidth=2, color="black"),
)
ax.text((x[peaks[1]] + x[peaks[-2]]) / 2, 0.98 + 0.03, f"$\\Delta \\,t_{{FP}}$", ha="center", va="bottom", clip_on=False, fontsize=14)

# ν_FSB
ax.annotate(
    "",
    xy=(x[peaks[-1]], 0.525), xytext=(x[peaks[-2]], 0.525),
    arrowprops=dict(arrowstyle="<->", linewidth=2, color="black"),
)
ax.text((x[peaks[-1]] + x[peaks[-2]]) / 2, 0.525 + 0.01, f"$\\Delta \\,t_{{FSR}}$", ha="center", va="bottom", clip_on=False, fontsize=14)

# FWHM
ax.annotate(
    "",
    xy=(-1.9, 0.49), xytext=(-2.3, 0.49),
    arrowprops=dict(arrowstyle="<->", linewidth=2, color="black"),
)
ax.text((-2.3 - 1.9) / 2, 0.49 - 0.03, f"$\\mathrm{{FWHM}}$", ha="center", va="top", fontsize=14, clip_on=False)



# set axis invisible
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.set_xticks([])
ax.set_yticks([])

# Leave room for the numbers below y=0
fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.22)

plt.savefig("FP/Laser/Modes.pdf", dpi=300, bbox_inches="tight", transparent=True, pad_inches=0.01)