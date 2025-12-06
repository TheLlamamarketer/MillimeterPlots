from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from PIL import Image, ImageSequence


mode = "circular"  # 'linear' or 'circular'
result_line = True
name_prefix = "Amplitude"  # Optional: add custom suffix like "_change", "_1", "_test", etc. Leave empty for default naming


# === PARAMETERS ===
a = 1/2 if mode == "circular" else 1/np.sqrt(2)

A1_start = 2 * a
A1_end = 0 * a      # Set different from A1_start to vary amplitude over time
A2_start = 0 * a
A2_end = 2 * a      # Set different from A2_start to vary amplitude over time

b = 0.6                  # spatial / temporal frequency scaling
theta = 2 * np.pi / 2  # phase difference between the two waves

# Handedness: 1 for LHS, -1 for RHS
handedness_1 = 1   # first wave
handedness_2 = -1  # second wave

loops_per_run = 5 # how many times the wave cycles over the full z range
vary_theta = False # whether to vary theta over time
vary_amplitude = True # whether to vary amplitude over time (from A_start to A_end)

# Animation timing: approximately 4 seconds per full rotation
seconds_per_rotation = 4
total_duration = loops_per_run * seconds_per_rotation  # Total animation duration in seconds
fps = 20  # Slightly lower FPS for smaller files

frames = int(fps * total_duration)

end_z = 4

# Rendering simplification to speed up and shrink output
rcParams['path.simplify'] = True
rcParams['path.simplify_threshold'] = 0.6
rcParams['agg.path.chunksize'] = 10000

# Sampling density for curves (lower = faster/smaller, still smooth)
n_points = 300

# GIF optimization (post-processing)
GIF_PALETTE_COLORS = 128


# === TRANSPARENT GIF WRITER ===
class TransparentGifWriter(PillowWriter):
    """PillowWriter with proper disposal so old frames are cleared."""
    def finish(self):
        # Save with optimize to reduce file size; disposal keeps transparency clean
        self._frames[0].save(
            self.outfile,
            save_all=True,
            append_images=self._frames[1:],
            duration=int(1000 / self.fps),
            loop=0,
            disposal=2,
            optimize=True,
        )


def optimize_gif(path: str, colors: int = GIF_PALETTE_COLORS):
    """Quantize and optimize a saved GIF using a valid RGBA method (FASTOCTREE)."""
    try:
        im = Image.open(path)
        frames = []
        durations = []
        for frame in ImageSequence.Iterator(im):
            rgba = frame.convert("RGBA")
            # Quantize using FASTOCTREE (method=2) which supports RGBA
            q = rgba.quantize(colors=colors, method=Image.FASTOCTREE)
            frames.append(q)
            durations.append(frame.info.get("duration", int(1000 / fps)))
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            disposal=2,
            optimize=True,
            duration=durations[0] if durations else int(1000 / fps),
        )
    except Exception as e:
        print(f"GIF optimize skipped: {e}")


# === ARROW CLASSES ===
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

class Arrow2D(FancyArrowPatch):
    """Simple 2D arrow for 2D plots."""
    def __init__(self, x_start, y_start, x_end, y_end, *args, **kwargs):
        super().__init__((x_start, y_start), (x_end, y_end), *args, **kwargs)


# === FIELD DEFINITION ===
def field_components(z, t, A, phase_offset, handedness):
    """
    Return Ex(z,t), Ey(z,t) for given amplitude, phase offset and handedness.
    z : array
    t : scalar (frame time)
    """
    phase = 2 * np.pi * (b * z + t) + phase_offset
    Ex = A * np.sin(phase)               
    Ey = handedness * A * np.cos(phase)   
    return Ex, Ey


# === POLARIZATION ELLIPSE ===
phi = np.linspace(0, 2 * np.pi, 400)


# ============================================================================
# SIDE VIEW (2D: Field components vs z)
# ============================================================================
def create_side_view():
    print("Creating side view animation...")
    
    fig = plt.figure(figsize=(10, 6), dpi=140)
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    
    # Setup axes
    ax.set_xlim(0, end_z)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('auto')
    ax.set_axis_off()
    ax.grid(False)
    ax.axhline(0, color='black', lw=1, alpha=0.3)
    ax.axhline(1, color='black', lw=1, alpha=0.1, linestyle="dotted")
    ax.axhline(-1, color='black', lw=1, alpha=0.1, linestyle="dotted")
    
    # Lines - side view
    line_1,    = ax.plot([], [], lw=2, color="tab:blue", alpha=0.6, label="Wave 1")     
    line_2,    = ax.plot([], [], lw=2, color="tab:red", alpha=0.6, label="Wave 2")   
    
    # Add legend
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    
    line_1.set_zorder(1)
    line_2.set_zorder(1)
    
    Arrow_1 = None
    Arrow_2 = None
    Arrow_res = None
    
    def init():
        for ln in (line_1, line_2):
            ln.set_data([], [])
        return line_1, line_2
    
    def animate(frame):
        nonlocal Arrow_1, Arrow_2, Arrow_res
        
        z = np.linspace(0, end_z, n_points)
        t = frame / frames * loops_per_run
        
        if vary_theta:
            theta_now = 2 * np.pi * (frame / frames)
        else:
            theta_now = theta
        
        if vary_amplitude:
            progress = frame / frames
            A1_now = A1_start + (A1_end - A1_start) * progress
            A2_now = A2_start + (A2_end - A2_start) * progress
        else:
            A1_now = A1_start
            A2_now = A2_start
        
        if mode == "linear":
            Ex1 = A1_now * np.sin(2 * np.pi * (b * z + t))
            Ey1 = np.zeros_like(z)
            Ex2 = np.zeros_like(z)
            Ey2 = A2_now * np.sin(2 * np.pi * (b * z + t) + theta_now)
        else:
            Ex1, Ey1 = field_components(z, t, A1_now, 0, handedness_1)
            Ex2, Ey2 = field_components(z, t, A2_now, theta_now, handedness_2)
        
        
        # Side view: for linear, show Ex for wave 1, Ey for wave 2, Ey for result
        # For circular, show magnitude of each wave
        if mode == "linear":
            wave1_side = Ex1  # Wave 1 oscillates in x
            wave2_side = Ey2  # Wave 2 oscillates in y
        else:
            wave1_side = Ey1
            wave2_side = Ey2
        
        line_1.set_data(z, wave1_side)
        line_2.set_data(z, wave2_side)
        
        # Arrows at z=0
        for arrow in (Arrow_1, Arrow_2, Arrow_res):
            if arrow:
                arrow.remove()
        
        if mode == "linear":
            Ex10 = A1_now * np.sin(2 * np.pi * t)
            Ey10 = 0
            Ex20 = 0
            Ey20 = A2_now * np.sin(2 * np.pi * t + theta_now)
        else:
            Ex10, Ey10 = field_components(0, t, A1_now, 0, handedness_1)
            Ex20, Ey20 = field_components(0, t, A2_now, theta_now, handedness_2)

        
        if mode == "linear":
            wave1_0 = Ex10
            wave2_0 = Ey20
        else:
            wave1_0 = Ey10
            wave2_0 = Ey20
        
        # Blue arrow: Wave 1
        Arrow_1 = Arrow2D(0, 0, 0, wave1_0, mutation_scale=15, lw=2, arrowstyle="-|>", color="tab:blue", alpha=0.9, zorder=5)
        ax.add_artist(Arrow_1)
        
        # Red arrow: Wave 2
        Arrow_2 = Arrow2D(0, 0, 0, wave2_0, mutation_scale=15, lw=2, arrowstyle="-|>", color="tab:red", alpha=0.9, zorder=5)
        ax.add_artist(Arrow_2)
        
        return line_1, line_2, Arrow_1, Arrow_2
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=50, blit=False)
    writer = TransparentGifWriter(fps=fps)
    
    name = f"{name_prefix}_{mode.capitalize()}_Polarization_Side.gif"
    anim.save(name, writer=writer, savefig_kwargs={"transparent": True, "facecolor": "none", "pad_inches": 0})
    # Post-optimization disabled to preserve transparency reliably
    print(f"Saved {name}")
    plt.close(fig)


# ============================================================================
# FRONT VIEW (2D: Ex vs Ey)
# ============================================================================
def create_front_view():
    print("Creating front view animation...")
    
    fig = plt.figure(figsize=(7, 6), dpi=140)
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    
    # Setup axes
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.grid(False)
    ax.axhline(0, color='black', lw=1, alpha=0.3)
    ax.axvline(0, color='black', lw=1, alpha=0.3)
    ax.axhline(1, color='black', lw=1, alpha=0.1, linestyle="dotted")
    ax.axhline(-1, color='black', lw=1, alpha=0.1, linestyle="dotted")
    ax.axvline(1, color='black', lw=1, alpha=0.1, linestyle="dotted")
    ax.axvline(-1, color='black', lw=1, alpha=0.1, linestyle="dotted")
    
    # Lines
    line_res,     = ax.plot([], [], lw=3, color="tab:green", alpha=0.9, label="Resultant")
    line_1,       = ax.plot([], [], lw=2, color="tab:blue", alpha=0.6, label="Wave 1")
    line_2,       = ax.plot([], [], lw=2, color="tab:red", alpha=0.6, label="Wave 2")
    ellipse_line, = ax.plot([], [], lw=1.5, color="gray", alpha=0.8, linestyle="dotted", zorder=0, label="Trajectory")
    
    # Add legend
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    
    line_1.set_zorder(1)
    line_2.set_zorder(1)
    line_res.set_zorder(2)
    
    Arrow_1 = None
    Arrow_2 = None
    Arrow_res = None
    
    def init():
        for ln in (line_res, line_1, line_2, ellipse_line):
            ln.set_data([], [])
        return line_res, line_1, line_2, ellipse_line
    
    def animate(frame):
        nonlocal Arrow_1, Arrow_2, Arrow_res
        
        t = frame / frames * loops_per_run
        
        if vary_theta:
            theta_now = 2 * np.pi * (frame / frames)
        else:
            theta_now = theta
        
        if vary_amplitude:
            progress = frame / frames
            A1_now = A1_start + (A1_end - A1_start) * progress
            A2_now = A2_start + (A2_end - A2_start) * progress
        else:
            A1_now = A1_start
            A2_now = A2_start
        
        if mode == "linear":
            Ex1_ellipse = A1_now * np.sin(phi)
            Ey1_ellipse = np.zeros_like(phi)
            line_1.set_data(Ex1_ellipse, Ey1_ellipse)
            
            Ex2_ellipse = np.zeros_like(phi)
            Ey2_ellipse = A2_now * np.sin(phi + theta_now)
            line_2.set_data(Ex2_ellipse, Ey2_ellipse)
        else:
            Ex1_ellipse = A1_now * np.sin(phi)
            Ey1_ellipse = handedness_1 * A1_now * np.cos(phi)
            line_1.set_data(Ex1_ellipse, Ey1_ellipse)
            
            Ex2_ellipse = A2_now * np.sin(phi + theta_now)
            Ey2_ellipse = handedness_2 * A2_now * np.cos(phi + theta_now)
            line_2.set_data(Ex2_ellipse, Ey2_ellipse)
        
        if result_line:
            Ex_res_ellipse = Ex1_ellipse + Ex2_ellipse
            Ey_res_ellipse = Ey1_ellipse + Ey2_ellipse
            line_res.set_data(Ex_res_ellipse, Ey_res_ellipse)
        else:
            line_res.set_data([], [])
        
        # Ellipse
        if mode == "linear":
            Ex1_ell = A1_now * np.sin(phi)
            Ey1_ell = np.zeros_like(phi)
            Ex2_ell = np.zeros_like(phi)
            Ey2_ell = A2_now * np.sin(phi + theta_now)
        else:
            Ex1_ell = A1_now * np.sin(phi)
            Ey1_ell = handedness_1 * A1_now * np.cos(phi)
            Ex2_ell = A2_now * np.sin(phi + theta_now)
            Ey2_ell = handedness_2 * A2_now * np.cos(phi + theta_now)
        
        Ex_ell = Ex1_ell + Ex2_ell
        Ey_ell = Ey1_ell + Ey2_ell
        ellipse_line.set_data(Ex_ell, Ey_ell)
        
        # Arrows
        for arrow in (Arrow_1, Arrow_2, Arrow_res):
            if arrow:
                arrow.remove()
        
        if mode == "linear":
            Ex10 = A1_now * np.sin(2 * np.pi * t)
            Ey10 = 0
            Ex20 = 0
            Ey20 = A2_now * np.sin(2 * np.pi * t + theta_now)
        else:
            Ex10, Ey10 = field_components(0, t, A1_now, 0, handedness_1)
            Ex20, Ey20 = field_components(0, t, A2_now, theta_now, handedness_2)
        
        Ex0_res = Ex10 + Ex20
        Ey0_res = Ey10 + Ey20
        
        Arrow_1 = Arrow2D(0, 0, Ex10, Ey10, mutation_scale=15, lw=2, arrowstyle="-|>", color="tab:blue", alpha=0.9, zorder=5)
        ax.add_artist(Arrow_1)
        
        Arrow_2 = Arrow2D(0, 0, Ex20, Ey20, mutation_scale=15, lw=2, arrowstyle="-|>", color="tab:red", alpha=0.9, zorder=5)
        ax.add_artist(Arrow_2)
        
        if result_line:
            Arrow_res = Arrow2D(0, 0, Ex0_res, Ey0_res, mutation_scale=20, lw=3, arrowstyle="-|>", color="tab:green", alpha=0.9, zorder=6)
            ax.add_artist(Arrow_res)
        
        return line_res, line_1, line_2, ellipse_line, Arrow_1, Arrow_2, Arrow_res
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=50, blit=False)
    writer = TransparentGifWriter(fps=fps)
    
    name = f"{name_prefix}_{mode.capitalize()}_Polarization_Front.gif"
    anim.save(name, writer=writer, savefig_kwargs={"transparent": True, "facecolor": "none", "pad_inches": 0})
    # Post-optimization disabled to preserve transparency reliably
    print(f"Saved {name}")
    plt.close(fig)


# ============================================================================
# NORMAL VIEW (3D)
# ============================================================================
def create_normal_view():
    print("Creating normal view animation...")
    
    fig = plt.figure(figsize=(7, 6), dpi=140)
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # Setup axes
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0, end_z)
    ax.view_init(elev=-45, azim=150, roll=-67)
    ax.set_axis_off()
    ax.grid(False)
    ax.plot([0, 0], [0, 0], [0, end_z], color="black", lw=1, alpha=0.3)
    ax.plot([-1, 1], [0, 0], [0, 0], color="black", lw=1, alpha=0.3)
    ax.plot([0, 0], [-1, 1], [0, 0], color="black", lw=1, alpha=0.3)
    ax.plot([-1, 1], [-1, -1], [0, 0], color="black", lw=1, alpha=0.1, linestyle="dotted")
    ax.plot([-1, 1], [1, 1], [0, 0], color="black", lw=1, alpha=0.1, linestyle="dotted")
    ax.plot([-1, -1], [-1, 1], [0, 0], color="black", lw=1, alpha=0.1, linestyle="dotted")
    ax.plot([1, 1], [-1, 1], [0, 0], color="black", lw=1, alpha=0.1, linestyle="dotted")
    
    # Lines
    line_res,     = ax.plot([], [], [], lw=3, color="tab:green", alpha=0.9, label="Resultant")
    line_1,       = ax.plot([], [], [], lw=2, color="tab:blue", alpha=0.6, label="Wave 1")
    line_2,       = ax.plot([], [], [], lw=2, color="tab:red", alpha=0.6, label="Wave 2")
    ellipse_line, = ax.plot([], [], [], lw=1.5, color="gray", alpha=0.8, linestyle="dotted", zorder=0, label="Trajectory")
    
    # Add legend
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    
    line_1.set_zorder(1)
    line_2.set_zorder(1)
    line_res.set_zorder(2)
    
    Arrow_1 = None
    Arrow_2 = None
    Arrow_res = None
    
    def init():
        for ln in (line_res, line_1, line_2, ellipse_line):
            ln.set_data([], [])
            ln.set_3d_properties([])
        return line_res, line_1, line_2, ellipse_line
    
    def animate(frame):
        nonlocal Arrow_1, Arrow_2, Arrow_res
        
        z = np.linspace(0, end_z, n_points)
        t = frame / frames * loops_per_run
        
        if vary_theta:
            theta_now = 2 * np.pi * (frame / frames)
        else:
            theta_now = theta
        
        if vary_amplitude:
            progress = frame / frames
            A1_now = A1_start + (A1_end - A1_start) * progress
            A2_now = A2_start + (A2_end - A2_start) * progress
        else:
            A1_now = A1_start
            A2_now = A2_start
        
        if mode == "linear":
            Ex1 = A1_now * np.sin(2 * np.pi * (b * z + t))
            Ey1 = np.zeros_like(z)
            Ex2 = np.zeros_like(z)
            Ey2 = A2_now * np.sin(2 * np.pi * (b * z + t) + theta_now)
        else:
            Ex1, Ey1 = field_components(z, t, A1_now, 0, handedness_1)
            Ex2, Ey2 = field_components(z, t, A2_now, theta_now, handedness_2)
        
        Ex_res = Ex1 + Ex2
        Ey_res = Ey1 + Ey2
        
        line_1.set_data(Ex1, Ey1)
        line_1.set_3d_properties(z)
        
        line_2.set_data(Ex2, Ey2)
        line_2.set_3d_properties(z)
        
        if result_line:
            line_res.set_data(Ex_res, Ey_res)
            line_res.set_3d_properties(z)
        else:
            line_res.set_data([], [])
            line_res.set_3d_properties([])
        
        # Ellipse
        if mode == "linear":
            Ex1_ell = A1_now * np.sin(phi)
            Ey1_ell = np.zeros_like(phi)
            Ex2_ell = np.zeros_like(phi)
            Ey2_ell = A2_now * np.sin(phi + theta_now)
        else:
            Ex1_ell = A1_now * np.sin(phi)
            Ey1_ell = handedness_1 * A1_now * np.cos(phi)
            Ex2_ell = A2_now * np.sin(phi + theta_now)
            Ey2_ell = handedness_2 * A2_now * np.cos(phi + theta_now)
        
        Ex_ell = Ex1_ell + Ex2_ell
        Ey_ell = Ey1_ell + Ey2_ell
        z_ell  = np.zeros_like(Ex_ell)
        
        ellipse_line.set_data(Ex_ell, Ey_ell)
        ellipse_line.set_3d_properties(z_ell)
        
        # Arrows
        for arrow in (Arrow_1, Arrow_2, Arrow_res):
            if arrow:
                arrow.remove()
        
        if mode == "linear":
            Ex10 = A1_now * np.sin(2 * np.pi * t)
            Ey10 = 0
            Ex20 = 0
            Ey20 = A2_now * np.sin(2 * np.pi * t + theta_now)
        else:
            Ex10, Ey10 = field_components(0, t, A1_now, 0, handedness_1)
            Ex20, Ey20 = field_components(0, t, A2_now, theta_now, handedness_2)
        
        Ex0_res = Ex10 + Ex20
        Ey0_res = Ey10 + Ey20
        
        Arrow_1 = Arrow3D([0, Ex10], [0, Ey10], [0, 0], mutation_scale=15, lw=2, arrowstyle="-|>", color="tab:blue", alpha=0.9, zorder=5)
        ax.add_artist(Arrow_1)
        
        Arrow_2 = Arrow3D([0, Ex20], [0, Ey20], [0, 0], mutation_scale=15, lw=2, arrowstyle="-|>", color="tab:red", alpha=0.9, zorder=5)
        ax.add_artist(Arrow_2)
        
        if result_line:
            Arrow_res = Arrow3D([0, Ex0_res], [0, Ey0_res], [0, 0], mutation_scale=20, lw=3, arrowstyle="-|>", color="tab:green", alpha=0.9, zorder=6)
            ax.add_artist(Arrow_res)
        
        return line_res, line_1, line_2, ellipse_line, Arrow_1, Arrow_2, Arrow_res
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=50, blit=False)
    writer = TransparentGifWriter(fps=fps)
    
    name = f"{name_prefix}_{mode.capitalize()}_Polarization_Normal.gif"
    anim.save(name, writer=writer, savefig_kwargs={"transparent": True, "facecolor": "none", "pad_inches": 0})
    # Post-optimization disabled to preserve transparency reliably
    print(f"Saved {name}")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print(f"Generating {mode} polarization animations...")
    print(f"Parameters: A1: {A1_start} to {A1_end}, A2: {A2_start} to {A2_end}, theta: {theta} rad, handedness: {handedness_1}, {handedness_2}")
    print()
    
    create_side_view()
    create_front_view()
    create_normal_view()
    
    print()
    print("All animations created successfully!")

