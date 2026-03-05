import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

# =========================
# PARAMETERS
# =========================
radius = 1.0
frames = 160

tilt_angle = 10.0          # degrees, 0 gives a circle, 90 gives a line
rv_exaggeration = 1.0      # boosts red and blue saturation
show_vr_text = False       # turn on if you want the number on screen

# Conventions
# vr > 0 means receding, red
# vr < 0 means approaching, blue

# =========================
# GEOMETRY
# =========================
tilt_rad = np.radians(tilt_angle)

# Orbit defined in x y plane, then tilted around x
# Display is the x z plane, so you see an oval
# Physics line of sight is along +y, so vr extremes occur at left and right

def pos_xyz(theta):
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = 0.0

    # tilt around x
    y_t = y * np.cos(tilt_rad) - z * np.sin(tilt_rad)
    z_t = y * np.sin(tilt_rad) + z * np.cos(tilt_rad)
    return x, y_t, z_t

def vel_xyz(theta):
    # derivative wrt theta, omega is irrelevant for color mapping
    dx = -radius * np.sin(theta)
    dy =  radius * np.cos(theta)
    dz = 0.0

    # tilt around x
    dy_t = dy * np.cos(tilt_rad) - dz * np.sin(tilt_rad)
    dz_t = dy * np.sin(tilt_rad) + dz * np.cos(tilt_rad)
    return dx, dy_t, dz_t

# line of sight for vr is +y in the tilted frame
los = np.array([0.0, 1.0, 0.0], dtype=float)

def vr(theta):
    vx, vy, vz = vel_xyz(theta)
    return float(vx * los[0] + vy * los[1] + vz * los[2])

# Precompute vr scale for stable colors
theta_dense = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
vr_dense = np.array([vr(t) for t in theta_dense], dtype=float)
vr_max = float(np.max(np.abs(vr_dense))) if np.max(np.abs(vr_dense)) > 0 else 1.0

def vr_to_color(v):
    x = rv_exaggeration * (v / vr_max)
    x = np.clip(x, -1.0, 1.0)

    # blue -> white -> red
    if x >= 0:
        return (1.0, 1.0 - x, 1.0 - x)
    x = -x
    return (1.0 - x, 1.0 - x, 1.0)

# =========================
# FIGURE
# =========================
fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.axis("off")

# =========================
# STATIC DRAWING
# =========================
# Orbit path in display plane x z
theta = np.linspace(0, 2 * np.pi, 600)
x_path = np.zeros_like(theta)
z_path = np.zeros_like(theta)
for i, t in enumerate(theta):
    x, y, z = pos_xyz(t)
    x_path[i] = x
    z_path[i] = z

ax.plot(x_path, z_path, linestyle="dashed", color=(0.7, 0.7, 0.7), alpha=0.5, lw=1.8)

# Central star
ax.scatter([0.0], [0.0], s=140, color="gold", alpha=0.9, zorder=10)

# Helper axes through the star
# Horizontal line z = 0 corresponds to max |vr| locations for this setup
ax.plot([-1.4, 1.4], [0.0, 0.0], color="white", alpha=0.25, lw=1.0)

# Mark special points
# Right side is max receding, left side is max approaching
special_points = [
    (0.0,  (1.0, 0.3, 0.3)),  # theta = 0, x = +r, vr max positive, red
    (np.pi, (0.3, 0.3, 1.0)), # theta = pi, x = -r, vr max negative, blue
    (0.5 * np.pi, (1.0, 1.0, 1.0)),     # vr ~ 0, top
    (1.5 * np.pi, (1.0, 1.0, 1.0)),     # vr ~ 0, bottom
]
for t, col in special_points:
    x, y, z = pos_xyz(t)
    ax.scatter([x], [z], s=70, color=col, alpha=0.95, zorder=9)

# Optional vr readout
info = ax.text(
    0.02, 0.96, "",
    transform=ax.transAxes,
    color="white",
    fontsize=12
)

# Animated planet
planet = ax.scatter([], [], s=520, color="white", zorder=20)

def init():
    planet.set_offsets(np.empty((0, 2)))
    info.set_text("")
    return planet, info

def animate(i):
    t = 2 * np.pi * i / frames

    x, y, z = pos_xyz(t)
    v = vr(t)
    col = vr_to_color(v)

    planet.set_offsets(np.array([[x, z]]))
    planet.set_color([col])

    if show_vr_text:
        info.set_text(f"vr (arb): {v:+.3f}")
    else:
        info.set_text("")

    return planet, info

anim = FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=frames,
    interval=50,
    blit=False
)

# =========================
# SAVE GIF
# =========================
raw_gif = "orbit_raw.gif"
anim.save(raw_gif, writer="pillow", fps=20, dpi=180)
print(f"Saved {raw_gif}")

# =========================
# POST PROCESS FOR TRANSPARENCY
# =========================
infile = raw_gif
outfile = "orbit_transparent.gif"

im = Image.open(infile)
frames_list = []

# Use a tolerance for background removal so small compression artifacts still get removed
bg_tol = 8  # 0..255

# First pass: make all frames transparent and find global bounding box
transparent_frames = []
x0_global, y0_global = float('inf'), float('inf')
x1_global, y1_global = 0, 0

for i in range(im.n_frames):
    im.seek(i)
    frame = im.convert("RGBA")
    arr = np.array(frame)

    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3]

    mask_bg = (rgb[..., 0] <= bg_tol) & (rgb[..., 1] <= bg_tol) & (rgb[..., 2] <= bg_tol)
    alpha[mask_bg] = 0
    arr[..., 3] = alpha

    frame = Image.fromarray(arr, mode="RGBA")
    transparent_frames.append(frame)
    
    # Update global bounding box
    nonzero = np.argwhere(arr[..., 3] > 0)
    if nonzero.size > 0:
        y0, x0 = nonzero.min(axis=0)
        y1, x1 = nonzero.max(axis=0) + 1
        x0_global = min(x0_global, x0)
        y0_global = min(y0_global, y0)
        x1_global = max(x1_global, x1)
        y1_global = max(y1_global, y1)

# Add padding and create final bbox
pad = 10
x0_global = max(0, int(x0_global) - pad)
y0_global = max(0, int(y0_global) - pad)
x1_global = min(transparent_frames[0].width, int(x1_global) + pad)
y1_global = min(transparent_frames[0].height, int(y1_global) + pad)
bbox = (x0_global, y0_global, x1_global, y1_global)

# Second pass: crop all frames with the global bbox
for frame in transparent_frames:
    cropped = frame.crop(bbox)
    frames_list.append(cropped)

frames_list[0].save(
    outfile,
    save_all=True,
    append_images=frames_list[1:],
    loop=0,
    duration=im.info.get("duration", 50),
    disposal=2
)

print(f"Saved {outfile}")


