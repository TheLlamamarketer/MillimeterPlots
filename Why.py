from matplotlib import pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # side-effect import for 3D

# === PARAMETERS ===
A_y   = 0.5    # amplitude of y-component
A_z   = 1.0    # amplitude of z-component
b     = 0.8    # spatial frequency scaling
frames = 100   # number of animation frames

# === FIGURE & AXIS SETUP ===
fig = plt.figure(figsize=(7, 6), dpi=180)
axis = fig.add_subplot(111, projection='3d')
fig.patch.set_alpha(0.0)

axis.set_xlim(0, 4)
axis.set_ylim(-1.2, 1.2)
axis.set_zlim(-1.2, 1.2)
axis.view_init(elev=20, azim=200)

# Turn off axis box and grid, draw reference x-axis
axis.set_axis_off()
axis.grid(False)
axis.plot([0, 4], [0, 0], [0, 0], color='black', lw=1)

# === STATIC POLARIZATION ELLIPSE AT x=0 ===
phi = np.linspace(0, 2 * np.pi, 400)
y_ellipse = A_y * np.sin(phi)
z_ellipse = A_z * np.cos(phi)

#axis.plot(
#    np.zeros_like(phi), y_ellipse, z_ellipse,
#    linestyle='dashed', color='gray', alpha=0.7, zorder=2
#)

# Vertical z-axis line at x=0 for reference
#axis.plot([0, 0], [0, 0], [-1.5, 1.5],
#          color='gray', linestyle='dotted', lw=0.8)

# === FIELDS PROPAGATING ALONG x ===
line, = axis.plot([], [], [], lw=3, color='tab:purple')  # elliptic (resultant) wave

# Re-added lines for linear polarization components
line_y, = axis.plot([], [], [], lw=2, color='tab:red', alpha=0.8)   # y-only (z=0)
line_z, = axis.plot([], [], [], lw=2, color='tab:blue', alpha=0.8)  # z-only (y=0)

# quivers for instantaneous decomposition at x=0
quiver_y = None   # component along y
quiver_z = None   # component along z
quiver_res = None # resultant vector

def init():
    """Initialize empty lines and no quivers."""
    for ln in (line, line_y, line_z):
        ln.set_data([], [])
        ln.set_3d_properties([])
    return line, line_y, line_z

def animate(i):
    global quiver_y, quiver_z, quiver_res

    # Wave along x, moving due to time-dependent phase
    x = np.linspace(4, 0, 1000)
    phase = - 2 * np.pi * i / frames * 2  # temporal phase
    arg = b * 2 * np.pi * x - phase

    # Elliptically polarized wave (components in y and z)
    y = A_y * np.sin(arg)
    z = A_z * np.sin(arg)

    # Set resultant (elliptic) wave
    line.set_data(x, y)
    line.set_3d_properties(z)
    
    line.set_data([], [])
    line.set_3d_properties([])

    # Set linear polarization component lines
    # y-only line: z = 0
    line_y.set_data(x, y)
    line_y.set_3d_properties(np.zeros_like(z))
    # z-only line: y = 0
    line_z.set_data(x, np.zeros_like(y))
    line_z.set_3d_properties(z)

    # Remove previous arrows
    for q in (quiver_y, quiver_z, quiver_res):
        if q is not None:
            q.remove()

    # Instantaneous field at x = 0
    y_val = A_y * np.sin(-phase)
    z_val = A_z * np.sin(-phase)

    # Component vectors from origin
    quiver_y = axis.quiver(
        0, 0, 0,
        0, y_val, 0,
        color='tab:red', arrow_length_ratio=0.3, linewidths=2
    )
    quiver_z = axis.quiver(
        0, 0, 0,
        0, 0, z_val,
        color='tab:blue', arrow_length_ratio=0.3, linewidths=2
    )

    # Resultant vector (sum of y and z components)
    quiver_res = axis.quiver(
        0, 0, 0,
        0, 0, 0,
        color='tab:purple', arrow_length_ratio=0.3, linewidths=2.2
    )

    return line, line_y, line_z, quiver_y, quiver_z, quiver_res

anim = FuncAnimation(
    fig, animate, init_func=init,
    frames=frames, interval=50, blit=False
)


anim.save(
    'line2_raw.gif',
    writer='pillow',
    fps=20,
    dpi=180,
)




from PIL import Image, ImageChops
import numpy as np

infile  = "line2_raw.gif"
outfile = "line2_transparent.gif"

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