from dataclasses import dataclass, field, replace
import csv
from itertools import product
import os
import shutil
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from numba import njit
from scipy.integrate import quad
from scipy.interpolate import griddata
from scipy.optimize import brentq, newton
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

@dataclass
class GeometryConfig:
    total_length: float = 0.05
    diameter: float = 330e-6
    n_points: int = 201

    displacement: float = 0.0085
    straight_len: float = 0.005
    
    half_model: bool = False


@dataclass
class MaterialConfig:
    density: float = 2200.0
    young_modulus: float = 73e9


@dataclass
class ContactConfig:
    contact_stiffness: float = 1e5
    friction: float = 0.2
    tangential_damping_ratio: float = 1.0
    restitution: float = 0.1
    impulse_friction: float = 0.2
    obstacle_radius: float = 0.002
    obtacles_locations: Sequence[tuple[float, float]] = field(
        default_factory=lambda: [(0.04, -1), (0.18, 1)]
    )
    obstacles: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-0.002, 0.022],
                [0.004, 0.039],
                
            ],
            dtype=np.float64,
        )
    )


@dataclass
class TimeConfig:
    n_steps: int = 3_000_000
    snap_every: int = 5_000
    chunk_size: int = 50_000
    time_speedup: float = 0.5


@dataclass
class DampingConfig:
    zeta_global: float = 0.4
    zeta_axial: float = 0.25
    zeta_bend: float = 1.0


@dataclass
class FailureConfig:
    tensile_strength: float = 1.0e9
    color_gamma: float = 0.6


@dataclass
class OutputConfig:
    preview_geometry: bool = True
    show_final_state: bool = True
    show_energy: bool = False
    show_animation: bool = True

    save_animation: bool = False
    animation_file: str = "fiber_verlet.mp4"
    animation_fps: int = 20
    animation_bitrate: int = 1800
    animation_dpi: int = 90
    animation_interval_ms: int = 40
    animation_stride: int = 1

    show_energy_in_animation: bool = True
    show_info_panel: bool = True

    run_zeta_sweep: bool = False
    zeta_sweep_factors: tuple[float, ...] = (0.01, 0.1, 0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 4.0, 10.0)
    
    run_obstacle_sweep: bool = False
    obstacle_range: Sequence[tuple[float, float]] = field(
        default_factory=lambda: [
            (0.04, 0.1),  
            (0.075, 0.25),
        ]
    )
    obstacle_sweep_refine_passes: int = 0
    obstacle_sweep_use_cache: bool = True
    obstacle_sweep_save_cache: bool = True
    obstacle_sweep_cache_file: str = "obstacle_sweep_cache_330.csv"
    obstacle_plot_gamma: float = 0.55


@dataclass
class FiberConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    damping: DampingConfig = field(default_factory=DampingConfig)
    failure: FailureConfig = field(default_factory=FailureConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class SimulationInputs:
    R0: np.ndarray
    m: np.ndarray
    dt: float
    l0: np.ndarray
    ks: float
    kb: float
    fixed: np.ndarray
    R_fixed: np.ndarray
    eta: float
    c_s_damp: float
    c_b_damp: float
    zeta_c: float




# ============================================================
# Derived helpers
# ============================================================

def section_properties(diameter: float):
    area = np.pi * (diameter / 2.0) ** 2
    inertia = np.pi * (diameter / 2.0) ** 4 / 4.0
    return area, inertia


def curve_shape(u):
    return  3*u**2 - 2*u**3

def curve_shape_derivative(u):
    return 6*(u - u**2)

def build_centerline_functions(cfg: FiberConfig):
    g = cfg.geometry

    mid_length = g.total_length - 2 * g.straight_len
    if mid_length <= 0.0:
        raise ValueError("straight_len must be smaller than total_length / 2")

    min_mid_arc = quad(lambda u: abs(g.displacement * curve_shape_derivative(u)), 0.0, 1.0, limit=2000)[0]
    if min_mid_arc > mid_length:
        raise ValueError(
            "displacement is too large for the available middle arc length; reduce displacement or increase total_length"
        )

    def mid_arc_length(y_scale):
        return quad(
            lambda u: np.sqrt((g.displacement * curve_shape_derivative(u)) ** 2 + y_scale**2),
            0.0,
            1.0,
            limit=2000,
        )[0]

    y_scale = brentq(lambda s: mid_arc_length(s) - mid_length, 0.0, mid_length)
    curve_arc_length = mid_arc_length(y_scale)

    t_start = g.straight_len / g.total_length
    t_end = 1.0 - g.straight_len / g.total_length

    def u_of_t(t):
        return (t - t_start) / (t_end - t_start)

    x = lambda t: np.where(
        t < t_start,
        0.0,
        np.where(
            t <= t_end,
            g.displacement * curve_shape(u_of_t(t)),
            g.displacement * curve_shape(1.0),
        ),
    )

    y = lambda t: np.where(
        t < t_start,
        t * g.total_length,
        np.where(
            t <= t_end,
            y_scale * u_of_t(t) + g.straight_len,
            y_scale + (t - t_end) * g.total_length + g.straight_len,
        ),
    )

    dx = lambda t: np.where(
        t < t_start,
        0.0,
        np.where(
            t <= t_end,
            g.displacement * curve_shape_derivative(u_of_t(t)) * (g.total_length / mid_length),
            0.0,
        ),
    )

    dy = lambda t: np.where(
        t < t_start,
        g.total_length,
        np.where(
            t <= t_end,
            y_scale * (g.total_length / mid_length),
            g.total_length,
        ),
    )

    info = {
        "curve_arc_length": curve_arc_length,
        "mid_length": mid_length,
        "y_scale": y_scale,
        "t_start": t_start,
        "t_end": t_end,
    }
    return x, y, dx, dy, info


def curve_length(x, y, target_length, dx=None, dy=None):
    def finite_diff(f, t):
        h = np.cbrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(t))
        return (f(t + h) - f(t - h)) / (2.0 * h)

    def speed(t):
        dx_dt = finite_diff(x, t) if dx is None else dx(t)
        dy_dt = finite_diff(y, t) if dy is None else dy(t)
        return np.sqrt(dx_dt**2 + dy_dt**2)

    def residual(t):
        return quad(speed, 0.0, t, limit=200)[0] - target_length

    x0 = max(1e-8, 0.8 * target_length)

    try:
        t1 = newton(residual, x0=x0, fprime=speed, maxiter=100)
    except RuntimeError:
        t_hi = max(1e-8, target_length)
        while residual(t_hi) < 0.0:
            t_hi *= 2.0
            if t_hi > 1e6:
                raise RuntimeError("Failed to bracket arc-length root in curve_length")
        t1 = brentq(residual, 0.0, t_hi)

    return t1


def parametric_lengths(x, y, t1, n_points, total_length):
    n_segments = n_points - 1
    tol = 1e-12

    if n_segments <= 0:
        raise ValueError("n_points must be at least 2")

    def chord_length(ta, tb):
        return np.hypot(x(tb) - x(ta), y(tb) - y(ta))

    def place_prefix(chord_target):
        ts = [0.0]
        for _ in range(n_segments - 1):
            ta = ts[-1]
            remaining = chord_length(ta, t1)
            if remaining < chord_target - tol * max(1.0, chord_target):
                raise ValueError("Chord length exceeds remaining curve length")
            root_fun = lambda tb: chord_length(ta, tb) - chord_target
            ts.append(brentq(root_fun, ta, t1))
        return np.array(ts)

    def last_segment_residual(chord_target):
        try:
            ts = place_prefix(chord_target)
        except ValueError:
            return -chord_target
        return chord_length(ts[-1], t1) - chord_target

    d_lo = 1e-12
    r_lo = last_segment_residual(d_lo)
    if r_lo <= 0.0:
        raise RuntimeError("Failed to bracket chord-spacing root at lower bound")

    if n_segments == 1:
        return np.array([0.0, t1]), chord_length(0.0, t1)

    # Physical upper bound: one segment cannot exceed the full end-to-end chord.
    d_hi = max(chord_length(0.0, t1) * (1.0 - 1e-12), 10.0 * d_lo)
    r_hi = last_segment_residual(d_hi)


    if r_hi > 0.0:
        raise RuntimeError("Failed to bracket chord-spacing root within physical chord bounds")

    chord_target = brentq(last_segment_residual, d_lo, d_hi)

    for _ in range(8):
        try:
            ts = place_prefix(chord_target)
            break
        except ValueError:
            chord_target *= 1.0 - 1e-10
    else:
        raise RuntimeError("Failed to place equal-chord points after root solve")

    ts = np.append(ts, t1)
    return ts, chord_target


def build_initial_rod(cfg: FiberConfig):
    x, y, dx, dy, curve_info = build_centerline_functions(cfg)
    t1 = curve_length(x, y, cfg.geometry.total_length, dx=dx, dy=dy)
    
    if cfg.geometry.half_model:
        t1 *= 0.5
        cfg.geometry.n_points = cfg.geometry.n_points//2 +1
    
    ts, chord_length = parametric_lengths(
        x=x,
        y=y,
        t1=t1,
        n_points=cfg.geometry.n_points,
        total_length=cfg.geometry.total_length,
    )

    positions = np.column_stack([x(ts), y(ts)])
    return positions, ts, chord_length, curve_info


def build_obstacles(cfg: FiberConfig, tangential_obstacle_pos=None):
    centers = np.asarray(cfg.contact.obstacles, dtype=np.float64)

    if tangential_obstacle_pos is not None:
        x, y, dx, dy, _ = build_centerline_functions(cfg)
        arr = np.asarray(tangential_obstacle_pos, dtype=np.float64)
        t1 = curve_length(x, y, cfg.geometry.total_length, dx=dx, dy=dy)
        t = arr[:, 0]/t1
        directions = -arr[:, 1]
        h = np.column_stack([x(t), y(t)])
        tx, ty = dx(t), dy(t)
        
        norm = np.maximum(np.hypot(tx, ty), 1e-15)
        nh = np.column_stack((-ty/norm, tx/norm)) * np.sign(directions[:, None])

        offset = 0.5 * cfg.geometry.diameter + cfg.contact.obstacle_radius + 1e-12
        centers = h + offset * nh

    radii = np.full((centers.shape[0], 1), cfg.contact.obstacle_radius, dtype=np.float64)
        
    obstacles = np.hstack([centers, radii])

    obstacles_effective = obstacles.copy()
    obstacles_effective[:, 2] += 0.5 * cfg.geometry.diameter
    return obstacles, obstacles_effective


def build_lumped_masses(R0, density, area):
    segment_lengths = np.linalg.norm(np.diff(R0, axis=0), axis=1)
    h = np.mean(segment_lengths)

    masses = np.ones(R0.shape[0]) * density * area * h
    masses[0] *= 0.5
    masses[-1] *= 0.5
    return masses, segment_lengths, h


def compute_time_step(cfg: FiberConfig, h, masses, area, inertia):
    E = cfg.material.young_modulus
    rho = cfg.material.density
    kc = cfg.contact.contact_stiffness

    c_axial = np.sqrt(E / rho)
    c_bending = np.sqrt(E * inertia / (rho * area))

    dt_axial = 0.2 * h / c_axial
    dt_bending = 0.05 * h**2 / (np.pi**2 * c_bending)
    dt_contact = 0.05 * np.sqrt(np.min(masses) / kc)

    dt = cfg.time.time_speedup * min(dt_axial, dt_bending, dt_contact)
    limits = {
        "dt_axial": dt_axial,
        "dt_bending": dt_bending,
        "dt_contact": dt_contact,
    }
    return dt, limits


def restitution_to_damping_ratio(e):
    if e <= 0.0:
        return 1.0
    if e >= 1.0:
        return 0.0
    loge = abs(np.log(e))
    return loge / np.sqrt(np.pi**2 + loge**2)


def compute_damping_coefficients(cfg: FiberConfig, h, masses, area, inertia):
    E = cfg.material.young_modulus
    rho = cfg.material.density

    ks_continuum = E * area
    kb_continuum = E * inertia

    ks = ks_continuum / h
    kb = kb_continuum / h**3

    c_bending_wave = np.sqrt(E * inertia / (rho * area))
    omega1 = (1.875104068711961**2) * c_bending_wave / cfg.geometry.total_length**2

    eta = 2.0 * cfg.damping.zeta_global * omega1

    m_segment = 0.5 * np.mean(masses)
    m_bend = np.mean(masses)

    c_axial = 2.0 * cfg.damping.zeta_axial * np.sqrt(ks * m_segment)
    c_bending = 2.0 * cfg.damping.zeta_bend * np.sqrt(kb * m_bend)

    return ks, kb, eta, c_axial, c_bending


# ============================================================
# Plot helpers
# ============================================================

def ribbon_quads(R, thickness):
    half = 0.5 * thickness
    n = R.shape[0]

    tangents = np.zeros_like(R)
    tangents[1:-1] = R[2:] - R[:-2]
    tangents[0] = R[1] - R[0]
    tangents[-1] = R[-1] - R[-2]

    lengths = np.sqrt(tangents[:, 0] ** 2 + tangents[:, 1] ** 2)
    lengths = np.maximum(lengths, 1e-15)

    normals = np.empty_like(R)
    normals[:, 0] = -tangents[:, 1] / lengths
    normals[:, 1] = tangents[:, 0] / lengths

    left = R + half * normals
    right = R - half * normals

    quads = np.empty((n - 1, 4, 2), dtype=R.dtype)
    quads[:, 0, :] = left[:-1]
    quads[:, 1, :] = left[1:]
    quads[:, 2, :] = right[1:]
    quads[:, 3, :] = right[:-1]
    return quads


def fiber_cmap():
    cmap = LinearSegmentedColormap.from_list(
        "risk_map",
        ["#1a9850", "#CBCB46", "#d73027"],
    ).copy()
    cmap.set_over("purple")
    return cmap


def failure_norm(cfg: FiberConfig):
    return PowerNorm(gamma=cfg.failure.color_gamma, vmin=0.0, vmax=1.0, clip=False)


def add_obstacles(ax, obstacles):
    for j in range(obstacles.shape[0]):
        circle = plt.Circle(
            (obstacles[j, 0], obstacles[j, 1]),
            obstacles[j, 2],
            color="orange",
            alpha=0.5,
            label="obstacle" if j == 0 else None,
        )
        ax.add_patch(circle)


def add_fixed_nodes(ax, fixed_positions):
    if fixed_positions.size:
        ax.scatter(fixed_positions[:, 0], fixed_positions[:, 1], color="red", label="fixed")


def set_equal_data_limits(ax, x_arrays, y_arrays, pad_fraction=0.05):
    xmin = min(np.min(x) for x in x_arrays)
    xmax = max(np.max(x) for x in x_arrays)
    ymin = min(np.min(y) for y in y_arrays)
    ymax = max(np.max(y) for y in y_arrays)

    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    half_span = 0.5 * max(xmax - xmin, ymax - ymin)
    pad = pad_fraction * max(2.0 * half_span, 1e-12)

    ax.set_xlim(xmid - half_span - pad, xmid + half_span + pad)
    ax.set_ylim(ymid - half_span - pad, ymid + half_span + pad)
    ax.set_aspect("equal", adjustable="box")


def plot_geometry_preview(R0, fixed, R_fixed, obstacles, cfg: FiberConfig):
    fig, ax = plt.subplots(figsize=(8.5, 7))

    preview = PolyCollection(
        ribbon_quads(R0, cfg.geometry.diameter),
        facecolors="tab:blue",
        edgecolors="none",
        label="fiber",
    )
    ax.add_collection(preview)

    add_fixed_nodes(ax, R_fixed[fixed])
    add_obstacles(ax, obstacles)
    set_equal_data_limits(ax, [R0[:, 0]], [R0[:, 1]])

    ax.set_title("Initial geometry")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_final_state(R0, R, fixed, R_fixed, obstacles, failure_index_final, cfg: FiberConfig):
    cmap = fiber_cmap()
    norm = failure_norm(cfg)

    fig, ax = plt.subplots(figsize=(8.5, 7))

    fiber = PolyCollection(
        ribbon_quads(R, cfg.geometry.diameter),
        cmap=cmap,
        norm=norm,
        edgecolors="none",
    )
    fiber.set_array(failure_index_final)

    ax.plot(R0[:, 0], R0[:, 1], "--", label="initial", alpha=0.5)
    ax.add_collection(fiber)
    add_fixed_nodes(ax, R_fixed[fixed])
    add_obstacles(ax, obstacles)
    set_equal_data_limits(ax, [R0[:, 0], R[:, 0]], [R0[:, 1], R[:, 1]])

    cbar = plt.colorbar(fiber, ax=ax, pad=0.01)
    cbar.set_label("Failure index $(\\sigma / \\sigma_{\\mathrm{ult}})$", rotation=270, labelpad=15)

    ax.set_title("Final state")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_energy_history(step_list, energy_hist, dt):
    if len(step_list) < 2:
        return

    time = (step_list + 1) * dt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, energy_hist[1:, 2], label="Kinetic")
    ax.plot(time, energy_hist[1:, 3], label="Potential")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def format_time_value(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds:.3f} s"


def create_animation(
    snapshots,
    energy_hist,
    step_list,
    failure_index,
    R0,
    fixed,
    R_fixed,
    obstacles,
    dt,
    cfg: FiberConfig,
):
    cmap = fiber_cmap()
    norm = failure_norm(cfg)

    total_length = np.sum(np.linalg.norm(np.diff(snapshots, axis=1), axis=2), axis=1)

    fig = plt.figure(figsize=(11.5, 7))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[4.8, 1.6],
        height_ratios=[1.0, 1.0],
        wspace=0.15,
        hspace=0.25,
    )

    ax_main = fig.add_subplot(gs[:, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, 1])

    ax_info.axis("off")
    info_text = ax_info.text(
        0.0,
        1.0,
        "",
        transform=ax_info.transAxes,
        va="top",
        ha="left",
        family="monospace",
        animated=True,
    )

    x_arrays = [R0[:, 0]] + [S[:, 0] for S in snapshots]
    y_arrays = [R0[:, 1]] + [S[:, 1] for S in snapshots]
    set_equal_data_limits(ax_main, x_arrays, y_arrays)

    ax_main.set_title("Fiber dynamics")
    ax_main.grid(True)
    ax_main.plot(R0[:, 0], R0[:, 1], "--", alpha=0.35, label="initial")
    add_obstacles(ax_main, obstacles)
    add_fixed_nodes(ax_main, R_fixed[fixed])

    fiber = PolyCollection([], cmap=cmap, norm=norm, edgecolors="none", animated=True)
    ax_main.add_collection(fiber)

    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_main, pad=0.01)
    cbar.set_label("Failure index $(\\sigma / \\sigma_{\\mathrm{ult}})$", rotation=270, labelpad=15)

    time = (step_list + 1) * dt
    ax_energy.set_title("Energy evolution")
    ax_energy.set_xlabel("Time (s)")
    ax_energy.set_ylabel("Energy (J)")
    ax_energy.set_yscale("log")
    ax_energy.grid(True)

    kinetic_line, = ax_energy.plot([], [], label="Kinetic")
    potential_line, = ax_energy.plot([], [], label="Potential")
    time_marker = ax_energy.axvline(0.0, linestyle="--")
    ax_energy.legend()

    if len(time) > 0:
        ax_energy.set_xlim(time[0], time[-1])
        positive = energy_hist[energy_hist > 0.0]
        if positive.size:
            ax_energy.set_ylim(0.8 * positive.min(), 1.2 * positive.max())

    def init():
        fiber.set_verts([])
        fiber.set_array(np.empty(0, dtype=R0.dtype))
        info_text.set_text("")
        kinetic_line.set_data([], [])
        potential_line.set_data([], [])
        if len(time) > 0:
            time_marker.set_xdata([time[0], time[0]])
        return fiber, info_text, kinetic_line, potential_line, time_marker

    def update(frame):
        Ri = snapshots[frame]
        fiber.set_verts(ribbon_quads(Ri, cfg.geometry.diameter))
        fiber.set_array(failure_index[frame])

        Es, Eb, Ek, Ep = energy_hist[frame]
        step = step_list[frame]
        t_elapsed = (step + 1) * dt

        info_text.set_text(
            f"step   = {step}\n"
            f"time   = {format_time_value(t_elapsed)}\n"
            f"length = {total_length[frame] * 1e3:.3f} mm\n"
            f"FImax  = {failure_index[frame].max():.3f}\n"
            f"Es     = {Es:.3e} J\n"
            f"Eb     = {Eb:.3e} J\n"
            f"Ek     = {Ek:.3e} J\n"
            f"Ep     = {Ep:.3e} J"
        )

        if cfg.output.show_energy_in_animation:
            kinetic_line.set_data(time[1: frame + 1], energy_hist[1: frame + 1, 2])
            potential_line.set_data(time[1: frame + 1], energy_hist[1: frame + 1, 3])
            time_marker.set_xdata([time[frame], time[frame]])

        return fiber, info_text, kinetic_line, potential_line, time_marker

    ani = FuncAnimation(
        fig,
        update,
        frames=range(0, len(snapshots), max(1, cfg.output.animation_stride)),
        init_func=init,
        interval=cfg.output.animation_interval_ms,
        blit=True,
        repeat=True,
        cache_frame_data=False,
    )

    return fig, ani


def resolve_ffmpeg_path():
    ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if ffmpeg_path:
        return ffmpeg_path

    env_override = os.environ.get("FFMPEG_PATH")
    if env_override and os.path.isfile(env_override):
        return env_override

    if os.name == "nt":
        fallback_dirs = [
            r"C:\Tools\ffmpeg-8.0.1-essentials_build\bin",
        ]
        for folder in fallback_dirs:
            candidate = os.path.join(folder, "ffmpeg.exe")
            if os.path.isfile(candidate):
                return candidate

    return None


def save_animation(ani, cfg: FiberConfig):
    if not cfg.output.save_animation:
        return

    ffmpeg_path = resolve_ffmpeg_path()
    if ffmpeg_path:
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path

    if animation.writers.is_available("ffmpeg"):
        writer = FFMpegWriter(
            fps=cfg.output.animation_fps,
            bitrate=cfg.output.animation_bitrate,
        )
        ani.save(cfg.output.animation_file, writer=writer, dpi=cfg.output.animation_dpi)
        print(f"Saved animation to {cfg.output.animation_file}")
    else:
        print("ffmpeg not found. Skipping MP4 export.")


# ============================================================
# Forces and energies
# ============================================================

@njit(cache=True)
def compute_forces(
    R,
    V,
    l0,
    ks,
    kb,
    kc,
    zeta_c,
    zeta_t,
    mu,
    eta,
    c_s_damp,
    c_b_damp,
    m,
    P,
    fixed,
):
    F = np.zeros_like(R)
    N = R.shape[0]

    for i in range(N):
        for j in range(P.shape[0]):
            rx = R[i, 0] - P[j, 0]
            ry = R[i, 1] - P[j, 1]
            d = np.sqrt(rx * rx + ry * ry)

            delta = P[j, 2] - d
            if delta > 0.0:
                if d > 1e-12:
                    nx = rx / d
                    ny = ry / d
                else:
                    nx = 0.0
                    ny = 1.0

                tx = -ny
                ty = nx

                vn = V[i, 0] * nx + V[i, 1] * ny
                vt = V[i, 0] * tx + V[i, 1] * ty

                c_n = 2.0 * zeta_c * np.sqrt(kc * m[i])
                c_t = 2.0 * zeta_t * np.sqrt(kc * m[i])

                fn = kc * delta
                if vn < 0.0:
                    fn -= c_n * vn

                if fn < 0.0:
                    fn = 0.0

                ft_trial = -c_t * vt
                ft_limit = mu * fn

                if ft_trial > ft_limit:
                    ft = ft_limit
                elif ft_trial < -ft_limit:
                    ft = -ft_limit
                else:
                    ft = ft_trial

                F[i, 0] += fn * nx + ft * tx
                F[i, 1] += fn * ny + ft * ty

    for i in range(N - 1):
        dx = R[i + 1, 0] - R[i, 0]
        dy = R[i + 1, 1] - R[i, 1]
        s = np.sqrt(dx * dx + dy * dy)
        if s < 1e-12:
            s = 1e-12

        ux = dx / s
        uy = dy / s

        f_spring = ks * (s - l0[i])

        dvx = V[i + 1, 0] - V[i, 0]
        dvy = V[i + 1, 1] - V[i, 1]
        v_rel = dvx * ux + dvy * uy

        f_damp = c_s_damp * v_rel

        fx = (f_spring + f_damp) * ux
        fy = (f_spring + f_damp) * uy

        F[i, 0] += fx
        F[i, 1] += fy
        F[i + 1, 0] -= fx
        F[i + 1, 1] -= fy

    for i in range(N - 2):
        qx = R[i + 2, 0] - 2.0 * R[i + 1, 0] + R[i, 0]
        qy = R[i + 2, 1] - 2.0 * R[i + 1, 1] + R[i, 1]

        qvx = V[i + 2, 0] - 2.0 * V[i + 1, 0] + V[i, 0]
        qvy = V[i + 2, 1] - 2.0 * V[i + 1, 1] + V[i, 1]

        fx = kb * qx + c_b_damp * qvx
        fy = kb * qy + c_b_damp * qvy

        F[i, 0] += -fx
        F[i, 1] += -fy
        F[i + 1, 0] += 2.0 * fx
        F[i + 1, 1] += 2.0 * fy
        F[i + 2, 0] += -fx
        F[i + 2, 1] += -fy

    g = 9.81
    for i in range(N):
        F[i, 1] -= m[i] * g

    for i in range(N):
        F[i, 0] -= eta * m[i] * V[i, 0]
        F[i, 1] -= eta * m[i] * V[i, 1]

        if fixed[i]:
            F[i, 0] = 0.0
            F[i, 1] = 0.0

    return F


@njit(cache=True)
def energies(R, l0, ks, kb, V, m):
    N = R.shape[0]
    E_stretch = 0.0
    E_bend = 0.0
    E_kinetic = 0.0

    for i in range(N - 1):
        dx = R[i + 1, 0] - R[i, 0]
        dy = R[i + 1, 1] - R[i, 1]
        s = np.sqrt(dx * dx + dy * dy)
        if s < 1e-12:
            s = 1e-12

        E_stretch += 0.5 * ks * (s - l0[i]) ** 2

    for i in range(N - 2):
        qx = R[i + 2, 0] - 2.0 * R[i + 1, 0] + R[i, 0]
        qy = R[i + 2, 1] - 2.0 * R[i + 1, 1] + R[i, 1]
        E_bend += 0.5 * kb * (qx**2 + qy**2)

    for i in range(N):
        E_kinetic += 0.5 * m[i] * (V[i, 0] ** 2 + V[i, 1] ** 2)

    E_potential = E_stretch + E_bend
    return E_stretch, E_bend, E_kinetic, E_potential


@njit(cache=True)
def resolve_node_circle_contacts(R_old, R, V, P, fixed, m, mu_imp):
    N = R.shape[0]
    eps = 1e-14

    for i in range(N):
        if fixed[i]:
            continue

        x0 = R_old[i, 0]
        y0 = R_old[i, 1]
        x1 = R[i, 0]
        y1 = R[i, 1]

        dx = x1 - x0
        dy = y1 - y0

        best_s = 2.0
        best_j = -1

        a = dx * dx + dy * dy
        if a > eps:
            for j in range(P.shape[0]):
                cx = P[j, 0]
                cy = P[j, 1]
                rad = P[j, 2]

                ox = x0 - cx
                oy = y0 - cy

                b = 2.0 * (ox * dx + oy * dy)
                c = ox * ox + oy * oy - rad * rad

                disc = b * b - 4.0 * a * c
                if disc >= 0.0:
                    sq = np.sqrt(disc)
                    s1 = (-b - sq) / (2.0 * a)
                    if 0.0 <= s1 <= 1.0 and s1 < best_s:
                        best_s = s1
                        best_j = j

        if best_j >= 0:
            cx = P[best_j, 0]
            cy = P[best_j, 1]
            rad = P[best_j, 2]

            hx = x0 + best_s * dx
            hy = y0 + best_s * dy

            rx = hx - cx
            ry = hy - cy
            d = np.sqrt(rx * rx + ry * ry)

            if d > eps:
                nx = rx / d
                ny = ry / d
            else:
                nx = 0.0
                ny = 1.0

            tx = -ny
            ty = nx

            R[i, 0] = cx + (rad + 1e-12) * nx
            R[i, 1] = cy + (rad + 1e-12) * ny

            vn = V[i, 0] * nx + V[i, 1] * ny
            vt = V[i, 0] * tx + V[i, 1] * ty

            Jn = 0.0
            if vn < 0.0:
                Jn = m[i] * (-vn)
                V[i, 0] -= vn * nx
                V[i, 1] -= vn * ny

            Jt_trial = -m[i] * vt
            Jt_limit = mu_imp * Jn

            if Jt_trial > Jt_limit:
                Jt = Jt_limit
            elif Jt_trial < -Jt_limit:
                Jt = -Jt_limit
            else:
                Jt = Jt_trial

            V[i, 0] += (Jt / m[i]) * tx
            V[i, 1] += (Jt / m[i]) * ty

        for j in range(P.shape[0]):
            cx = P[j, 0]
            cy = P[j, 1]
            rad = P[j, 2]

            rx = R[i, 0] - cx
            ry = R[i, 1] - cy
            d = np.sqrt(rx * rx + ry * ry)

            if d < rad:
                if d > eps:
                    nx = rx / d
                    ny = ry / d
                else:
                    nx = 0.0
                    ny = 1.0

                tx = -ny
                ty = nx

                R[i, 0] = cx + (rad + 1e-12) * nx
                R[i, 1] = cy + (rad + 1e-12) * ny

                vn = V[i, 0] * nx + V[i, 1] * ny
                vt = V[i, 0] * tx + V[i, 1] * ty

                Jn = 0.0
                if vn < 0.0:
                    Jn = m[i] * (-vn)
                    V[i, 0] -= vn * nx
                    V[i, 1] -= vn * ny

                Jt_trial = -m[i] * vt
                Jt_limit = mu_imp * Jn

                if Jt_trial > Jt_limit:
                    Jt = Jt_limit
                elif Jt_trial < -Jt_limit:
                    Jt = -Jt_limit
                else:
                    Jt = Jt_trial

                V[i, 0] += (Jt / m[i]) * tx
                V[i, 1] += (Jt / m[i]) * ty

    return R, V


# ============================================================
# Time stepping
# ============================================================

@njit(cache=True)
def _verlet_chunk(
    R,
    V,
    l0,
    ks,
    kb,
    kc,
    zeta_c,
    zeta_t,
    mu,
    eta,
    c_s_damp,
    c_b_damp,
    m,
    fixed,
    R_fixed,
    dt,
    snap_every,
    step_offset,
    n_chunk,
    max_snaps,
    snapshots,
    energy_hist,
    step_list,
    ns,
    P,
    quiet_count,
    v_tol,
    min_steps_for_convergence,
    mu_imp,
    quiet_needed=5,
):
    N = R.shape[0]
    converged = False
    steps_done = 0

    for local_step in range(n_chunk):
        step = step_offset + local_step
        steps_done = local_step + 1
        vmax = 0.0

        F = compute_forces(
            R, V, l0, ks, kb, kc, zeta_c, zeta_t, mu, eta, c_s_damp, c_b_damp, m, P, fixed
        )

        for i in range(N):
            V[i, 0] += 0.5 * dt * F[i, 0] / m[i]
            V[i, 1] += 0.5 * dt * F[i, 1] / m[i]

        R_old = R.copy()

        for i in range(N):
            R[i, 0] += dt * V[i, 0]
            R[i, 1] += dt * V[i, 1]
            if fixed[i]:
                R[i, 0] = R_fixed[i, 0]
                R[i, 1] = R_fixed[i, 1]

        R, V = resolve_node_circle_contacts(R_old, R, V, P, fixed, m, mu_imp)

        F = compute_forces(
            R, V, l0, ks, kb, kc, zeta_c, zeta_t, mu, eta, c_s_damp, c_b_damp, m, P, fixed
        )

        for i in range(N):
            V[i, 0] += 0.5 * dt * F[i, 0] / m[i]
            V[i, 1] += 0.5 * dt * F[i, 1] / m[i]

            if fixed[i]:
                V[i, 0] = 0.0
                V[i, 1] = 0.0

            vmag = np.sqrt(V[i, 0] ** 2 + V[i, 1] ** 2)
            if vmag > vmax:
                vmax = vmag

        if step % snap_every == 0:
            if ns < max_snaps:
                Es, Eb, Ek, Ep = energies(R, l0, ks, kb, V, m)
                snapshots[ns] = R
                energy_hist[ns, 0] = Es
                energy_hist[ns, 1] = Eb
                energy_hist[ns, 2] = Ek
                energy_hist[ns, 3] = Ep
                step_list[ns] = step
                ns += 1

            if step >= min_steps_for_convergence and vmax < v_tol:
                quiet_count += 1
            else:
                quiet_count = 0

            if quiet_count >= quiet_needed:
                converged = True
                break

    return R, V, ns, converged, quiet_count, steps_done


def verlet_simulation(
    R0,
    m,
    dt,
    l0,
    ks,
    kb,
    fixed,
    R_fixed,
    cfg: FiberConfig,
    P_effective,
    eta,
    c_s_damp,
    c_b_damp,
    zeta_c,
):
    R = R0.copy()
    V = np.zeros_like(R0)

    n_steps = int(cfg.time.n_steps)
    snap_every = int(cfg.time.snap_every)
    chunk_size = int(cfg.time.chunk_size)

    max_snaps = n_steps // snap_every + 1
    N = R0.shape[0]

    snapshots = np.empty((max_snaps, N, 2), dtype=R0.dtype)
    energy_hist = np.empty((max_snaps, 4), dtype=R0.dtype)
    step_list = np.empty(max_snaps, dtype=np.int64)

    h = np.mean(l0)
    v_tol = 1e-2 * h / max(snap_every * dt, 1e-30)

    with tqdm(total=n_steps, desc="Simulating", unit="steps") as pbar:
        quiet_count = 0
        step_offset = 0
        ns = 0

        while step_offset < n_steps:
            n_chunk = min(chunk_size, n_steps - step_offset)

            R, V, ns, converged, quiet_count, steps_done = _verlet_chunk(
                R=R,
                V=V,
                l0=l0,
                ks=ks,
                kb=kb,
                kc=cfg.contact.contact_stiffness,
                zeta_c=zeta_c,
                zeta_t=cfg.contact.tangential_damping_ratio,
                mu=cfg.contact.friction,
                eta=eta,
                c_s_damp=c_s_damp,
                c_b_damp=c_b_damp,
                m=m,
                fixed=fixed,
                R_fixed=R_fixed,
                dt=dt,
                snap_every=snap_every,
                step_offset=step_offset,
                n_chunk=n_chunk,
                max_snaps=max_snaps,
                snapshots=snapshots,
                energy_hist=energy_hist,
                step_list=step_list,
                ns=ns,
                P=P_effective,
                quiet_count=quiet_count,
                v_tol=v_tol,
                min_steps_for_convergence=5 * snap_every,
                mu_imp=cfg.contact.impulse_friction,
            )

            pbar.update(steps_done)
            step_offset += steps_done

            if converged:
                break

    return R, V, snapshots[:ns], energy_hist[:ns], step_list[:ns]


# ============================================================
# Diagnostics
# ============================================================

def tensile_failure_index_segments(R, l0, E, diameter, sigma_ult):
    n = R.shape[0]
    if n < 3:
        return np.zeros(max(n - 1, 0), dtype=R.dtype)

    dR = R[1:] - R[:-1]
    seg_len = np.sqrt(dR[:, 0] ** 2 + dR[:, 1] ** 2)
    seg_len = np.maximum(seg_len, 1e-15)

    eps_axial = (seg_len - l0) / np.maximum(l0, 1e-15)

    seg_dir = dR / seg_len[:, None]
    dot = np.sum(seg_dir[:-1] * seg_dir[1:], axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)

    ds_mid = 0.5 * (seg_len[:-1] + seg_len[1:])
    kappa_mid = theta / np.maximum(ds_mid, 1e-15)

    kappa_node = np.zeros(n, dtype=R.dtype)
    kappa_node[1:-1] = kappa_mid
    kappa_node[0] = kappa_mid[0]
    kappa_node[-1] = kappa_mid[-1]

    kappa_seg = 0.5 * (kappa_node[:-1] + kappa_node[1:])

    eps_bend = 0.5 * diameter * kappa_seg
    eps_tension = np.maximum(eps_axial + eps_bend, 0.0)

    sigma_tension = E * eps_tension
    return sigma_tension / max(sigma_ult, 1e-15)


def compute_failure_history(snapshots, l0, cfg: FiberConfig):
    failure = np.empty((snapshots.shape[0], snapshots.shape[1] - 1), dtype=snapshots.dtype)
    for k in range(snapshots.shape[0]):
        failure[k] = tensile_failure_index_segments(
            snapshots[k],
            l0=l0,
            E=cfg.material.young_modulus,
            diameter=cfg.geometry.diameter,
            sigma_ult=cfg.failure.tensile_strength,
        )
    return failure


def run_simulation(cfg: FiberConfig, P_effective, sim: SimulationInputs):
    return verlet_simulation(
        R0=sim.R0,
        m=sim.m,
        dt=sim.dt,
        l0=sim.l0,
        ks=sim.ks,
        kb=sim.kb,
        fixed=sim.fixed,
        R_fixed=sim.R_fixed,
        cfg=cfg,
        P_effective=P_effective,
        eta=sim.eta,
        c_s_damp=sim.c_s_damp,
        c_b_damp=sim.c_b_damp,
        zeta_c=sim.zeta_c,
    )



def _refine_axis(axis_values: np.ndarray, passes: int) -> np.ndarray:
    refined = np.array(axis_values, dtype=np.float64)
    for _ in range(max(0, int(passes))):
        mids = 0.5 * (refined[:-1] + refined[1:])
        refined = np.unique(np.concatenate([refined, mids]))
    return refined


def _param_key(values: Sequence[float]) -> str:
    return ",".join(f"{float(v):.10f}" for v in values)


def _load_obstacle_cache(cache_file: str, n_params: int) -> dict[str, float]:
    cache: dict[str, float] = {}
    if not os.path.isfile(cache_file):
        return cache

    with open(cache_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        param_cols = [f"p{i+1}" for i in range(n_params)]
        for row in reader:
            try:
                vals = [float(row[c]) for c in param_cols]
                tilt = float(row["tilt"])
            except (KeyError, ValueError):
                continue
            cache[_param_key(vals)] = tilt
    return cache


def _save_obstacle_cache(cache_file: str, cache: dict[str, float], n_params: int) -> None:
    param_cols = [f"p{i+1}" for i in range(n_params)]
    with open(cache_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([*param_cols, "tilt"])
        for key in sorted(cache.keys()):
            vals = [float(v) for v in key.split(",")]
            writer.writerow([*vals, float(cache[key])])


def _run_obstacle_candidate(cfg: FiberConfig, sim: SimulationInputs, sweep_param_locations):
    cfg_i = replace(
        cfg,
        contact=replace(cfg.contact, obtacles_locations=sweep_param_locations),
    )
    tangential_i = object_locations(cfg_i.contact, half_model=cfg_i.geometry.half_model)
    obstacles_plot_i, obstacles_effective_i = build_obstacles(cfg_i, tangential_obstacle_pos=tangential_i)

    R_i, _, snapshots_i, _, step_i = run_simulation(
        cfg=cfg_i,
        P_effective=obstacles_effective_i,
        sim=sim,
    )

    time_i = (step_i + 1) * sim.dt if len(step_i) else np.array([], dtype=np.float64)
    tilt_history_i = (
        np.array([end_straightness_score(S, n_end=10, pos=0) for S in snapshots_i], dtype=np.float64)
        if len(snapshots_i)
        else np.array([], dtype=np.float64)
    )

    return {
        "obstacle_param_locations": sweep_param_locations,
        "tangential_obstacle_pos": tangential_i,
        "obstacle_centers": obstacles_plot_i[:, :2].copy(),
        "R_final": R_i,
        "steps": step_i,
        "time": time_i,
        "end_tilt_score_final": end_straightness_score(R_i, n_end=10, pos=0),
        "end_tilt_score_history": tilt_history_i,
    }


def run_obstacle_sweep(cfg: FiberConfig, sim: SimulationInputs):
    results = []

    base_param_locations = list(cfg.contact.obtacles_locations)
    ranges = list(cfg.output.obstacle_range)
    if len(ranges) != len(base_param_locations):
        raise ValueError("output.obstacle_range must match number of contact.obtacles_locations")

    n_params = len(base_param_locations)
    n_samples_per_dim = 10
    axes = [np.linspace(lo, hi, n_samples_per_dim) for (lo, hi) in ranges]
    axes = [_refine_axis(ax, cfg.output.obstacle_sweep_refine_passes) for ax in axes]

    cache_file = os.path.abspath(cfg.output.obstacle_sweep_cache_file)
    cache = _load_obstacle_cache(cache_file, n_params) if cfg.output.obstacle_sweep_use_cache else {}
    cache_updated = False

    for t_values in product(*axes):
        sweep_param_locations = [
            (float(t_values[i]), base_param_locations[i][1])
            for i in range(n_params)
        ]
        key = _param_key([v[0] for v in sweep_param_locations])

        if key in cache:
            results.append(
                {
                    "obstacle_param_locations": sweep_param_locations,
                    "tangential_obstacle_pos": np.empty((0, 2), dtype=np.float64),
                    "obstacle_centers": np.empty((0, 2), dtype=np.float64),
                    "R_final": np.empty((0, 2), dtype=np.float64),
                    "steps": np.array([], dtype=np.int64),
                    "time": np.array([], dtype=np.float64),
                    "end_tilt_score_final": float(cache[key]),
                    "end_tilt_score_history": np.array([], dtype=np.float64),
                    "from_cache": True,
                }
            )
            continue

        res = _run_obstacle_candidate(cfg, sim, sweep_param_locations)
        res["from_cache"] = False
        results.append(res)
        cache[key] = float(res["end_tilt_score_final"])
        cache_updated = True

    # Ensure top candidates have full trajectory data for plotting overlays.
    if results:
        ranked = sorted(range(len(results)), key=lambda i: results[i]["end_tilt_score_final"])
        for idx in ranked[: min(8, len(ranked))]:
            if results[idx].get("from_cache", False):
                full = _run_obstacle_candidate(cfg, sim, results[idx]["obstacle_param_locations"])
                full["from_cache"] = False
                results[idx] = full

    if cfg.output.obstacle_sweep_save_cache and (cache_updated or (cfg.output.obstacle_sweep_use_cache and cache)):
        _save_obstacle_cache(cache_file, cache, n_params)

    print(f"Obstacle sweep cache: {cache_file} (entries={len(cache)})")
    return results


def plot_obstacle_sweep(results, R0, top_n=6, color_gamma=0.55):
    if not results:
        return

    n_params = len(results[0]["obstacle_param_locations"])
    p1_vals = np.array([r["obstacle_param_locations"][0][0] for r in results], dtype=np.float64)
    if n_params >= 2:
        p2_vals = np.array([r["obstacle_param_locations"][1][0] for r in results], dtype=np.float64)
        param_diff = np.abs(p2_vals - p1_vals)
    else:
        # 1D fallback: keep a non-negative distance-like quantity for ranking/plotting.
        param_diff = np.abs(p1_vals)

    tilt_scores = np.array([r["end_tilt_score_final"] for r in results], dtype=np.float64)

    p_span = max(float(np.max(param_diff) - np.min(param_diff)), 1e-12)
    t_span = max(float(np.max(tilt_scores) - np.min(tilt_scores)), 1e-12)
    p_norm = (param_diff - np.min(param_diff)) / p_span
    t_norm = (tilt_scores - np.min(tilt_scores)) / t_span

    # Higher is better. Keep tilt as the dominant term, with parameter difference as a weaker boost.
    tilt_good = 1.0 - t_norm
    tilt_weight = 2.0
    param_weight = 0.5
    combined_score = np.power(np.maximum(tilt_good, 1e-12), tilt_weight) * np.power(1.0 + p_norm, param_weight)

    ranked_tilt_idx = np.argsort(tilt_scores)
    ranked_combined_idx = np.argsort(combined_score)[::-1]
    top_idx = ranked_tilt_idx[: max(1, min(top_n, len(results)))]
    best_combined = results[int(ranked_combined_idx[0])]
    best_tilt = results[int(ranked_tilt_idx[0])]
    top = [results[int(i)] for i in top_idx]

    # Plot 1/2: tilt landscape and combined-score landscape.
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.3))
    ax_param, ax_combined, ax_shape = axes

    p1 = p1_vals.tolist()
    score = tilt_scores

    p2 = p2_vals.tolist()

    pts = np.column_stack((p1, p2))
    p1_lin = np.linspace(min(p1), max(p1), 220)
    p2_lin = np.linspace(min(p2), max(p2), 220)
    P1g, P2g = np.meshgrid(p1_lin, p2_lin)

    Z = griddata(pts, np.asarray(score), (P1g, P2g), method="cubic")
    if np.isnan(Z).any():
        Z_lin = griddata(pts, np.asarray(score), (P1g, P2g), method="linear")
        Z_near = griddata(pts, np.asarray(score), (P1g, P2g), method="nearest")
        Z = np.where(np.isnan(Z), Z_lin, Z)
        Z = np.where(np.isnan(Z), Z_near, Z)

    # Tilt score is physically non-negative; clamp interpolation artifacts.
    Z = np.maximum(Z, 0.0)

    z_valid = Z[np.isfinite(Z)]
    z_min = float(np.min(z_valid)) if z_valid.size else 0.0
    z_max = float(np.max(z_valid)) if z_valid.size else 1.0
    norm = PowerNorm(gamma=max(1e-3, float(color_gamma)), vmin=z_min, vmax=max(z_min + 1e-12, z_max))

    sc = ax_param.pcolormesh(P1g, P2g, Z, shading="auto", cmap="viridis_r", norm=norm)
    ax_param.scatter(p1, p2, c=score, s=16, cmap="viridis_r", norm=norm, edgecolors="k", linewidths=0.25)
    ax_param.set_ylabel("param 2 (curve coordinate)")

    C = griddata(pts, np.asarray(combined_score), (P1g, P2g), method="cubic")
    if np.isnan(C).any():
        C_lin = griddata(pts, np.asarray(combined_score), (P1g, P2g), method="linear")
        C_near = griddata(pts, np.asarray(combined_score), (P1g, P2g), method="nearest")
        C = np.where(np.isnan(C), C_lin, C)
        C = np.where(np.isnan(C), C_near, C)

    # Combined score is non-negative by construction; clamp interpolation undershoot.
    C = np.maximum(C, 0.0)

    c_valid = C[np.isfinite(C)]
    c_min = float(np.min(c_valid)) if c_valid.size else 0.0
    c_max = float(np.max(c_valid)) if c_valid.size else 1.0
    gamma_inv = 1.0 / max(1e-3, float(color_gamma))
    c_norm = PowerNorm(gamma=gamma_inv, vmin=c_min, vmax=max(c_min + 1e-12, c_max))

    sc_c = ax_combined.pcolormesh(P1g, P2g, C, shading="auto", cmap="plasma", norm=c_norm)
    ax_combined.scatter(p1, p2, c=combined_score, s=16, cmap="plasma", norm=c_norm, edgecolors="k", linewidths=0.25)
    ax_combined.set_xlabel("param 1 (curve coordinate)")
    ax_combined.set_ylabel("param 2 (curve coordinate)")
    ax_combined.set_title("Combined score landscape")
    ax_combined.grid(True)
    plt.colorbar(sc_c, ax=ax_combined, pad=0.01, label="combined score")

    ax_param.set_title("Obstacle parameter sweep")
    ax_param.set_xlabel("param 1 (curve coordinate)")
    ax_param.grid(True)
    plt.colorbar(sc, ax=ax_param, pad=0.01, label="final end tilt score (lower is better)")

    # Plot 3: final shape overlays for top candidates.
    ax_shape.plot(R0[:, 0], R0[:, 1], "--", color="0.35", label="initial")
    for r in top:
        params = ", ".join(f"{loc[0]:.3f}" for loc in r["obstacle_param_locations"])
        ax_shape.plot(r["R_final"][:, 0], r["R_final"][:, 1], label=f"p=[{params}]")

    ax_shape.set_title("Top final shapes")
    ax_shape.set_xlabel("x (m)")
    ax_shape.set_ylabel("y (m)")
    ax_shape.grid(True)
    ax_shape.set_aspect("equal", adjustable="box")
    ax_shape.legend(fontsize=7)

    print(
        "Best by combined score:",
        [tuple(v) for v in best_combined["obstacle_param_locations"]],
        f"(diff={abs(best_combined['obstacle_param_locations'][1][0] - best_combined['obstacle_param_locations'][0][0]) if n_params >= 2 else abs(best_combined['obstacle_param_locations'][0][0]):.3f})",
        f"-> final tilt score={best_combined['end_tilt_score_final']:.3f}",
        f", combined(mult)={np.max(combined_score):.3f}",
    )
    print(
        "Best by tilt only:",
        [tuple(v) for v in best_tilt["obstacle_param_locations"]],
        f"(diff={abs(best_tilt['obstacle_param_locations'][1][0] - best_tilt['obstacle_param_locations'][0][0]) if n_params >= 2 else abs(best_tilt['obstacle_param_locations'][0][0]):.3f})",
        f"-> final tilt score={best_tilt['end_tilt_score_final']:.3f}",
    )

    n_report = min(5, len(results))
    print(f"\nTop {n_report} by tilt only (absolute objective):")
    for rank, idx in enumerate(ranked_tilt_idx[:n_report], start=1):
        r = results[int(idx)]
        pvals = [loc[0] for loc in r["obstacle_param_locations"]]
        print(
            f"  {rank}. params={np.round(pvals, 4).tolist()} | "
            f"tilt={r['end_tilt_score_final']:.3f} | "
            f"diff={abs(pvals[1] - pvals[0]) if len(pvals) >= 2 else abs(pvals[0]):.3f} | "
            f"combined(mult)={combined_score[int(idx)]:.3f}"
        )

    plt.tight_layout()
    plt.show()



def run_zeta_global_sweep(
    cfg: FiberConfig,
    zeta_values: Sequence[float],
    sim_base: SimulationInputs,
    P_effective,
):
    results = []

    for zeta in zeta_values:
        cfg_i = replace(cfg, damping=replace(cfg.damping, zeta_global=float(zeta)))
        _, _, eta_i, _, _ = compute_damping_coefficients(
            cfg_i,
            h=np.mean(sim_base.l0),
            masses=sim_base.m,
            area=np.pi * (cfg.geometry.diameter / 2.0) ** 2,
            inertia=np.pi * (cfg.geometry.diameter / 2.0) ** 4 / 4.0,
        )

        sim_i = replace(sim_base, eta=eta_i)
        R_i, _, snapshots_i, energy_i, step_i = run_simulation(
            cfg=cfg_i,
            P_effective=P_effective,
            sim=sim_i,
        )

        time_i = (step_i + 1) * sim_base.dt if len(step_i) else np.array([], dtype=np.float64)
        max_disp_i = (
            np.max(np.linalg.norm(snapshots_i - sim_base.R0[None, :, :], axis=2), axis=1)
            if len(snapshots_i)
            else np.array([], dtype=np.float64)
        )

        results.append(
            {
                "zeta_global": float(zeta),
                "R_final": R_i,
                "snapshots": snapshots_i,
                "energy": energy_i,
                "steps": step_i,
                "time": time_i,
                "max_displacement": max_disp_i,
            }
        )

    return results


def plot_zeta_global_sweep(results, R0):
    if not results:
        return

    ref = results[np.argmin([abs(r["zeta_global"] - 1.0) for r in results])]["R_final"]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    ax_conv, ax_shape = axes

    for r in results:
        label = f"zeta_g={r['zeta_global']:.3g}"

        if len(r["time"]) and len(r["max_displacement"]):
            ax_conv.plot(r["time"], r["max_displacement"] * 1e3, label=label)

        d = np.linalg.norm(r["R_final"] - ref, axis=1)
        rms = np.sqrt(np.mean(d**2))
        dmax = np.max(d)
        steps_done = int(r["steps"][-1]) if len(r["steps"]) else 0
        print(
            f"zeta_global={r['zeta_global']:.3g}: "
            f"steps={steps_done}, "
            f"RMS(final-ref)={rms:.3e} m, "
            f"MAX(final-ref)={dmax:.3e} m"
        )

    ax_conv.set_title("Convergence history")
    ax_conv.set_xlabel("Time (s)")
    ax_conv.set_ylabel("Max node displacement from initial (mm)")
    ax_conv.grid(True)
    ax_conv.legend(fontsize=8)

    ax_shape.plot(R0[:, 0], R0[:, 1], "--", color="0.35", label="initial")
    for r in results:
        ax_shape.plot(r["R_final"][:, 0], r["R_final"][:, 1], label=f"z={r['zeta_global']:.3g}")

    ax_shape.set_title("Final shapes by zeta_global")
    ax_shape.set_xlabel("x (m)")
    ax_shape.set_ylabel("y (m)")
    ax_shape.grid(True)
    ax_shape.set_aspect("equal", adjustable="box")
    ax_shape.legend(fontsize=8)

    plt.tight_layout()
    plt.show()
    
    
    
    
def object_locations(parameters, half_model=False):
    
    list = []
    for parameters in parameters.obtacles_locations:
        list.append(parameters)
        if not half_model:
            list.append((1.0 - parameters[0], -parameters[1]))

    return list


def end_straightness_score(R, n_end=10, pos=0):
    n_end = int(max(3, min(n_end, R.shape[0] - 1)))

    dR = R[1:] - R[:-1]
    seg_len = np.linalg.norm(dR, axis=1)
    seg_len = np.maximum(seg_len, 1e-15)
    u = dR / seg_len[:, None]

    def mean_vertical_deviation(units):
        vertical_alignment = np.clip(np.abs(units[:, 1]), 0.0, 1.0)
        angle = np.mean(np.arccos(vertical_alignment))
        return 100.0 * angle / (0.5 * np.pi)

    vertical_deviation = mean_vertical_deviation(u[:n_end]) if pos == 0 else mean_vertical_deviation(u[-n_end:])
    return vertical_deviation




# ============================================================
# Main
# ============================================================

def main():
    cfg = FiberConfig()

    area, inertia = section_properties(cfg.geometry.diameter)


    R0, _, chord_length, _ = build_initial_rod(cfg)
    masses, l0, h = build_lumped_masses(R0, cfg.material.density, area)
    


    ks, kb, eta, c_s_damp, c_b_damp = compute_damping_coefficients(cfg, h, masses, area, inertia)
    dt, dt_limits = compute_time_step(cfg, h, masses, area, inertia)
    zeta_c = restitution_to_damping_ratio(cfg.contact.restitution)

    print(
        f"Time step limits: "
        f"dt_axial={dt_limits['dt_axial']:.3e}, "
        f"dt_bending={dt_limits['dt_bending']:.3e}, "
        f"dt_contact={dt_limits['dt_contact']:.3e}"
    )
    print(f"Using dt = {dt:.3e}")
    print(f"Mean segment length h = {h:.3e}")
    
    tangential_obstacle_pos = object_locations(
        cfg.contact,
        half_model=cfg.geometry.half_model,
    )
    
    obstacles_plot, obstacles_effective = build_obstacles(cfg, tangential_obstacle_pos)

    fixed = np.zeros(R0.shape[0], dtype=np.bool_)
    R_fixed = R0.copy()

    if cfg.geometry.half_model:
        fixed[-1] = True
        R_fixed[-1] = R0[-1]

    if cfg.output.preview_geometry:
        plot_geometry_preview(R0, fixed, R_fixed, obstacles_plot, cfg)
    
    sim_base = SimulationInputs(
        R0=R0,
        m=masses,
        dt=dt,
        l0=l0,
        ks=ks,
        kb=kb,
        fixed=fixed,
        R_fixed=R_fixed,
        eta=eta,
        c_s_damp=c_s_damp,
        c_b_damp=c_b_damp,
        zeta_c=zeta_c,
    )

    R, V, snapshots, energy_hist, step_list = run_simulation(
        cfg=cfg,
        P_effective=obstacles_effective,
        sim=sim_base,
    )

    failure_index = compute_failure_history(snapshots, l0, cfg)
    max_failure = float(failure_index.max()) if failure_index.size else 0.0
    print(f"Max tensile failure index sigma/sigma_ult = {max_failure:.3f}")
    print(f"Final end tilt score (0=vertical, 100=horizontal): {end_straightness_score(R):.2f}")

    if cfg.output.show_final_state and len(snapshots) > 0:
        plot_final_state(
            R0=R0,
            R=R,
            fixed=fixed,
            R_fixed=R_fixed,
            obstacles=obstacles_plot,
            failure_index_final=failure_index[-1],
            cfg=cfg,
        )

    if cfg.output.show_energy and len(snapshots) > 0:
        plot_energy_history(step_list, energy_hist, dt)

    if cfg.output.show_animation and len(snapshots) > 0:
        fig, ani = create_animation(
            snapshots=snapshots,
            energy_hist=energy_hist,
            step_list=step_list,
            failure_index=failure_index,
            R0=R0,
            fixed=fixed,
            R_fixed=R_fixed,
            obstacles=obstacles_plot,
            dt=dt,
            cfg=cfg,
        )
        save_animation(ani, cfg)
        plt.show()


        if cfg.output.run_obstacle_sweep:
            sweep_results = run_obstacle_sweep(
                cfg=cfg,
                sim=sim_base,
            )
            plot_obstacle_sweep(sweep_results, R0, color_gamma=cfg.output.obstacle_plot_gamma)





    if cfg.output.run_zeta_sweep:
        factors = np.asarray(cfg.output.zeta_sweep_factors, dtype=np.float64)
        zeta_values = np.unique(np.clip(cfg.damping.zeta_global * factors, 1e-9, np.inf))

        cfg_sweep = replace(
            cfg,
            output=replace(
                cfg.output,
                preview_geometry=False,
                show_final_state=False,
                show_energy=False,
                show_animation=False,
                save_animation=False,
            ),
        )

        print("\nRunning zeta_global sweep:", zeta_values)
        sweep_results = run_zeta_global_sweep(
            cfg=cfg_sweep,
            zeta_values=zeta_values,
            sim_base=sim_base,
            P_effective=obstacles_effective,
        )
        plot_zeta_global_sweep(sweep_results, R0)


if __name__ == "__main__":
    main()