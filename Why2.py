from re import A
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D import side-effect

# === PARAMETERS ===
A_L = 1.0   # amplitude of left-hand circular polarization
A_R = 1.0  # amplitude of right-hand circular polarization
b   = 0.8   # spatial frequency scaling
frames = 150

fig = plt.figure(figsize=(7, 6), dpi=180)
axis = fig.add_subplot(111, projection='3d')
axis.set_xlim(0, 4)
axis.set_ylim(-1.1, 1.1)
axis.set_zlim(-1.1, 1.1)
axis.view_init(elev=15, azim=210)

# reference line along x, y=z=0
axis.plot([0, 4], [0, 0], [0, 0], color='black', lw=1)

# --- Polarization curves at x=0, adjusted by amplitudes ---
phi = -1* np.linspace(0, 2 * np.pi, 400)

# LCP circle (blue)
y_L = A_L * np.sin(phi)
z_L = A_L * np.cos(phi)
axis.plot(np.zeros_like(phi), y_L, z_L,
          linestyle='dashed', color='tab:blue', alpha=0.7, zorder=2)

# RCP circle (red)
y_R = -A_R * np.sin(phi)
z_R = A_R * np.cos(phi)
axis.plot(np.zeros_like(phi), y_R, z_R,
          linestyle='dashed', color='tab:red', alpha=0.7, zorder=2)

# Resulting polarization ellipse (green)
y_res = (A_L - A_R) * np.sin(phi)
z_res = (A_L + A_R) * np.cos(phi)
axis.plot(np.zeros_like(phi), y_res, z_res,
          linestyle='solid', color='tab:green', lw=1.8, zorder=3)

# Axis line in z to emphasize the plane
axis.plot([0, 0], [0, 0], [-2, 2], color='tab:green', linestyle='dashed', lw=0.5)

axis.set_axis_off()
axis.grid(False)

# --- Fields propagating along x ---

line1, = axis.plot([], [], [], lw=3, color='tab:blue')  # LCP
line2, = axis.plot([], [], [], lw=3, color='tab:red')   # RCP

# quivers for instantaneous vectors
quiver_y = None
quiver_z = None
quiver_line = None
quiver_y_to_line = None
quiver_z_to_line = None


def init():
    line1.set_data([], [])
    line1.set_3d_properties([])
    line2.set_data([], [])
    line2.set_3d_properties([])
    return line1, line2


def animate(i):
    global quiver_y, quiver_z, quiver_line, quiver_y_to_line, quiver_z_to_line

    x = np.linspace(4, 0, 1000)
    phase = -2 * np.pi * i / frames * 2
    arg = b * 2 * np.pi * x - phase

    # Left-hand circular wave
    y1 = A_L * np.sin(arg)
    z1  = A_L * np.cos(arg)  # common z-phase

    # Right-hand circular wave (opposite y component)
    y2 = -A_R * np.sin(arg)
    z2 = A_R * np.cos(arg)  # common z-phase

    line1.set_data(x, y1)
    line1.set_3d_properties(z1)

    line2.set_data(x, y2)
    line2.set_3d_properties(z2)
    # Remove previous arrows
    for q in (quiver_y, quiver_z, quiver_line, quiver_y_to_line, quiver_z_to_line):
        if q is not None:
            q.remove()

    # --- Instantaneous polarization vectors at x=0 ---

    tip_y = np.array([0.0,  A_L * np.sin(-phase), A_L * np.cos(-phase)])  # LCP tip
    tip_z = np.array([0.0, -A_R * np.sin(-phase), A_R * np.cos(-phase)])  # RCP tip

    vec_addition = tip_y + tip_z               # resultant tip
    vec_y_to_line = vec_addition - tip_y       # arrow from LCP tip to resultant
    vec_z_to_line = vec_addition - tip_z       # arrow from RCP tip to resultant

    # Component arrows from origin
    quiver_y = axis.quiver(
        0, 0, 0,
        tip_y[0], tip_y[1], tip_y[2],
        color='tab:blue', arrow_length_ratio=0.25,
        linewidths=2
    )
    quiver_z = axis.quiver(
        0, 0, 0,
        tip_z[0], tip_z[1], tip_z[2],
        color='tab:red', arrow_length_ratio=0.25,
        linewidths=2
    )

    # Resultant arrow
    quiver_line = axis.quiver(
        0, 0, 0,
        vec_addition[0], vec_addition[1], vec_addition[2],
        color='tab:green', arrow_length_ratio=0.25,
        linewidths=2.2
    )

    # Small arrows showing vector addition (from tip_y and tip_z)
    quiver_y_to_line = axis.quiver(
        tip_y[0], tip_y[1], tip_y[2],
        vec_y_to_line[0], vec_y_to_line[1], vec_y_to_line[2],
        color='tab:orange', arrow_length_ratio=0.3,
        linewidths=1.5
    )
    quiver_z_to_line = axis.quiver(
        tip_z[0], tip_z[1], tip_z[2],
        vec_z_to_line[0], vec_z_to_line[1], vec_z_to_line[2],
        color='tab:orange', arrow_length_ratio=0.3,
        linewidths=1.5
    )

    return line1, line2, quiver_y, quiver_z, quiver_line, quiver_y_to_line, quiver_z_to_line


anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=50, blit=False)
anim.save('Circle1.gif', writer='pillow', fps=20)


from PIL import Image, ImageChops
import numpy as np

infile  = "Circle1.gif"
outfile = "Circle1_transparent.gif"

im = Image.open(infile)

frames = []
bg_color = np.array([255, 255, 255], dtype=np.uint8)  # assumed background

bbox = None

for i in range(im.n_frames):
    im.seek(i)
    frame = im.convert("RGBA")
    arr = np.array(frame)  # (H, W, 4)

    # Make background transparent: where RGB == bg_color → alpha = 0
    rgb = arr[..., :3]
    alpha = arr[..., 3]

    mask_bg = np.all(rgb == bg_color, axis=-1)
    alpha[mask_bg] = 0
    arr[..., 3] = alpha

    # Convert back to Pillow image
    frame = Image.fromarray(arr, mode="RGBA")

    # Determine common crop box from first frame (based on non-transparent pixels)
    if i == 0:
        nonzero = np.argwhere(arr[..., 3] > 0)
        if nonzero.size > 0:
            y0, x0 = nonzero.min(axis=0)   # row, col
            y1, x1 = nonzero.max(axis=0) + 1
            bbox = (x0, y0, x1, y1)
        else:
            # fallback: no visible pixels → keep whole frame
            bbox = (0, 0, frame.width, frame.height)

    frame = frame.crop(bbox)
    frames.append(frame)

# Save with disposal=2 so each frame clears previous one
frames[0].save(
    outfile,
    save_all=True,
    append_images=frames[1:],
    loop=0,
    duration=im.info.get("duration", 50),
    disposal=2,
    transparency=0  # index will be computed from first frame
)