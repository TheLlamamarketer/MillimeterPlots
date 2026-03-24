from itertools import product

import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar, minimize


@dataclass
class StaticCurveConfig:
    total_length: float = 1.0
    displacement: float = 0.1
    diameter: float = 0.01
    len_front: float = 0.1
    len_end: float = 0.1



@dataclass
class PegSpec:
    name: str
    side_sign: int       # +1 or -1
    radius: float
    y_min: float
    y_max: float
    
    
@dataclass
class PegPlacement:
    name: str
    side_sign: int
    radius: float
    y_contact: float
    center: np.ndarray


def s(u):
    return 6*u**5 - 15*u**4 + 10*u**3

def ds(u):
    return 30*u**4 - 60*u**3 + 30*u**2

def d2s(u):
    return 120*u**3 - 180*u**2 + 60*u


def middle_arc_length(Hm, d):
    return quad(
        lambda u: np.sqrt((d * ds(u)) ** 2 + Hm**2),
        0.0,
        1.0,
        limit=2000,
    )[0]


def middle_height(cfg: StaticCurveConfig, Lf=0.0, Le=0.0):
    L = cfg.total_length
    d = cfg.displacement

    Lm = L - Lf - Le
    if Lm <= 0.0:
        raise ValueError("len_front + len_end must be smaller than total_length")

    min_possible_arc = abs(d) * quad(lambda u: np.abs(ds(u)), 0.0, 1.0, limit=2000)[0]
    if min_possible_arc > Lm:
        raise ValueError(
            "displacement is too large for the available middle arc length. "
            "Reduce displacement or increase total_length / reduce straight sections."
        )

    def residual(Hm):
        return middle_arc_length(Hm, d) - Lm

    Hm = brentq(residual, 0.0, Lm)
    return Hm, Lm


def build_curve(cfg: StaticCurveConfig, Lf, Le):
    d = cfg.displacement

    Hm, Lm = middle_height(cfg, Lf, Le)
    y0 = 0.0
    y1 = Lf
    y2 = Lf + Hm
    y3 = Lf + Hm + Le

    def eta(y):
        return (y - y1) / Hm

    def x(y):
        y = np.asarray(y)
        out = np.empty_like(y, dtype=np.float64)

        mask_front = y < y1
        mask_mid = (y >= y1) & (y <= y2)
        mask_end = y > y2

        out[mask_front] = 0.0
        out[mask_mid] = d * s(eta(y[mask_mid]))
        out[mask_end] = d
        return out

    def dxdy(y):
        y = np.asarray(y)
        out = np.zeros_like(y, dtype=np.float64)

        mask_mid = (y >= y1) & (y <= y2)
        out[mask_mid] = d * ds(eta(y[mask_mid])) / Hm
        return out

    def d2xdy2(y):
        y = np.asarray(y)
        out = np.zeros_like(y, dtype=np.float64)

        mask_mid = (y >= y1) & (y <= y2)
        out[mask_mid] = d * d2s(eta(y[mask_mid])) / (Hm**2)
        return out

    def curvature_of_y(y):
        x1 = dxdy(y)
        x2 = d2xdy2(y)
        return x2 / (1.0 + x1**2) ** 1.5

    info = {
        "front_length": Lf,
        "end_length": Le,
        "middle_arc_length": Lm,
        "middle_height": Hm,
        "y_start": y0,
        "y_front_end": y1,
        "y_middle_end": y2,
        "y_end": y3,
        "x_end": d,
    }

    return x, dxdy, d2xdy2, curvature_of_y, info


def sample_curve(cfg: StaticCurveConfig, Le, Lf, n=1000):
    x, dxdy, d2xdy2, curvature_of_y, info = build_curve(cfg, Lf, Le)

    y = np.linspace(info["y_start"], info["y_end"], n)

    return {
        "y": y,
        "x": x(y),
        "dxdy": dxdy(y),
        "d2xdy2": d2xdy2(y),
        "kappa": curvature_of_y(y),
        "info": info,
    }


def arc_length_from_sample(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sum(np.sqrt(dx**2 + dy**2))

def mirror_curve(cfg: StaticCurveConfig, Le, Lf):
    x, dxdy, d2xdy2, curvature_of_y, info = build_curve(cfg, Lf, Le)

    def x_mirror(y):
        return cfg.diameter + 2 * cfg.displacement - x(y)

    def dxdy_mirror(y):
        return -dxdy(y)

    def d2xdy2_mirror(y):
        return -d2xdy2(y)

    def curvature_of_y_mirror(y):
        return -curvature_of_y(y)

    return x_mirror, dxdy_mirror, d2xdy2_mirror, curvature_of_y_mirror, info





def bending_energy(dxdy, curvature_of_y, ystart, yend):
    def integrand(y): 
        return curvature_of_y(y)**2 * np.sqrt(1.0 + dxdy(y)**2)
    
    energy, _ = quad(integrand, ystart, yend, limit=2000)
    return energy

def straightness_penalty(dxdy, ystart, yend):
    def integrand(y):
        return dxdy(y)**2
    
    penalty, _ = quad(integrand, ystart, yend, limit=2000)
    return penalty



def peg_center(x_fun, dxdy_fun, y_contact, r_eff, side_sign):
    slope = dxdy_fun(y_contact)
    norm = np.sqrt(1.0 + slope**2)
    normal = np.array([1.0, -slope]) / norm
    return np.array([x_fun(y_contact), y_contact]) + side_sign * r_eff * normal



def highest_constrained_point(x, dxdy, x_mirr, r_eff, Lf, Hm, n_scan=2000):
    y_min = Lf
    y_max = Lf + Hm

    def dist_point_to_mirror_curve(y_contact):
        c = peg_center(x, dxdy, y_contact, r_eff, side_sign=1)

        def d2(ym):
            dx_ = x_mirr(ym) - c[0]
            dy_ = ym - c[1]
            return dx_ * dx_ + dy_ * dy_

        res = minimize_scalar(d2, bounds=(y_min, y_max), method="bounded")
        return np.sqrt(res.fun), res.x

    def residual(y_contact):
        dmin, _ = dist_point_to_mirror_curve(y_contact)
        return dmin - r_eff

    # Scan to find the highest feasible point
    ys = np.linspace(y_min, y_max, n_scan)
    vals = np.array([residual(y) for y in ys])

    feasible = vals >= 0.0
    if not np.any(feasible):
        return None

    # Highest feasible sampled point
    i_last = np.where(feasible)[0][-1]
    y_lo = ys[i_last]

    # Refine if not at the top boundary
    if i_last < len(ys) - 1:
        y_hi = ys[i_last + 1]
        if residual(y_lo) * residual(y_hi) <= 0:
            y_high = brentq(residual, y_lo, y_hi, xtol=1e-9, rtol=1e-9, maxiter=200)
        else:
            y_high = y_lo
    else:
        y_high = y_lo

    return y_high



def build_pegs(y_contacts, side_signs, radii, x, dxdy, x_mirr, fiber_diameter, Lf, Le, Hm, clearance=0.0, n_pegs=None):
    if n_pegs is None:
        n_pegs = len(y_contacts)
    if n_pegs <= 0:
        raise ValueError("n_pegs must be >= 1.")

    expected_sides = np.array([(-1) ** (i + 1) for i in range(n_pegs)], dtype=np.int64)

    if side_signs is None:
        side_arr = expected_sides
    else:
        side_arr = np.asarray(side_signs, dtype=np.int64)

    radii_arr = np.asarray(radii, dtype=np.float64)
    if radii_arr.ndim == 0:
        radii_arr = np.full(n_pegs, float(radii_arr))

    y_min = Lf
    y_max_global = Lf + Hm
    centers = np.empty((n_pegs, 2), dtype=np.float64)
    prev_y = y_max_global

    for i in range(n_pegs):
        y_item = y_contacts[i]

        # Accept either one y value or a candidate list per peg.
        if np.isscalar(y_item):
            candidates = np.array([float(y_item)], dtype=np.float64)
        else:
            candidates = np.asarray(list(y_item), dtype=np.float64)
            if candidates.size == 0:
                raise ValueError(f"Peg {i} has no y candidates.")

        candidates = np.sort(candidates)[::-1]

        side = int(side_arr[i])
        radius = float(radii_arr[i])
        y_upper = min(prev_y, y_max_global)
        r_eff = 0.5 * fiber_diameter + clearance + radius

        # Keep positive-side pegs below their own mirror-clearance ceiling.
        if side > 0:
            y_clearance_max = highest_constrained_point(
                x=x,
                dxdy=dxdy,
                x_mirr=x_mirr,
                r_eff=r_eff,
                Lf=Lf,
                Hm=Hm,
            )
            if y_clearance_max is None:
                raise ValueError(f"No feasible constrained y exists for peg {i} with side=+1.")
            y_upper = min(y_upper, float(y_clearance_max))

        y_pick = None
        center_pick = None

        for y_try in candidates:
            if y_try < y_min or y_try > y_upper:
                continue
            y_pick = float(y_try)
            center_pick = peg_center(x, dxdy, y_pick, r_eff, side)
            break

        if y_pick is None:
            raise ValueError(
                f"No feasible y found for peg {i}. Required range: [{y_min}, {y_upper}]."
            )

        centers[i] = center_pick
        prev_y = y_pick

    return centers




def sweep_pegs(cfg, Lf, Le, side_signs, radii, n_scan=40, clearance=0.0):
    x, dxdy, d2xdy2, curvature_of_y, info = build_curve(cfg, Lf, Le)
    x_mirr, _, _, _, _ = mirror_curve(cfg, Le, Lf)

    y_grid = np.linspace(info["y_front_end"], info["y_middle_end"], n_scan)

    n_pegs = len(side_signs)
    results = []
    
    for y_tuple in product(y_grid, repeat=n_pegs):
        try:
            centers = build_pegs(
                y_contacts=y_tuple,
                side_signs=side_signs,
                radii=radii,
                x=x,
                dxdy=dxdy,
                x_mirr=x_mirr,
                fiber_diameter=cfg.diameter,
                Lf=Lf,
                Le=Le,
                Hm=info["middle_height"],
                clearance=clearance,
                n_pegs=n_pegs,
            )
            results.append((y_tuple, centers))
        except ValueError:
            continue

        


    



def plot_static_curve(cfg: StaticCurveConfig, n=1000):
    data = sample_curve(cfg, n=n)
    x = data["x"]
    y = data["y"]
    dxdy = data["dxdy"]
    kappa = data["kappa"]
    info = data["info"]

    L_num = arc_length_from_sample(x, y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    ax0, ax1, ax2 = axes

    ax0.plot(x, y, lw=2)
    ax0.axhline(info["y_front_end"], ls="--", alpha=0.5, label="front -> middle")
    ax0.axhline(info["y_middle_end"], ls="--", alpha=0.5, label="middle -> end")
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_title("Static curve x(y)")
    ax0.grid(True)
    ax0.legend()

    ax1.plot(y, dxdy, lw=2)
    ax1.set_xlabel("y")
    ax1.set_ylabel("dx/dy")
    ax1.set_title("Slope")
    ax1.grid(True)

    ax2.plot(y, kappa, lw=2)
    ax2.set_xlabel("y")
    ax2.set_ylabel("curvature")
    ax2.set_title("Curvature")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    print("---- curve info ----")
    for k, v in info.items():
        print(f"{k}: {v:.6f}")
    print(f"target total_length: {cfg.total_length:.6f}")
    print(f"sampled arc length : {L_num:.6f}")
    print(f"max |curvature|    : {np.max(np.abs(kappa)):.6f}")
    
    
def plot_geometry_with_constraint_point(cfg: StaticCurveConfig, peg_radius=0.03, clearance=0.0, n=2000):
    x_fun, dxdy_fun, _, _, info = build_curve(cfg, cfg.len_front, cfg.len_end)
    x_mirr_fun, dxdy_mirr_fun, _, _, _ = mirror_curve(cfg, cfg.len_end, cfg.len_front)

    y = np.linspace(info["y_start"], info["y_end"], n)
    x = x_fun(y)
    x_mirr = x_mirr_fun(y)

    res = highest_constrained_point(
        x=x_fun,
        dxdy=dxdy_fun,
        x_mirr=x_mirr_fun,
        peg_radius=peg_radius,
        fiber_diameter=cfg.diameter,
        Lf=info["y_front_end"],
        Hm=info["middle_height"],
        clearance=clearance,
        n_scan=2000,
    )

    fig, ax = plt.subplots(figsize=(7, 8))

    # centerlines
    ax.plot(x, y, lw=2, label="fiber 1")
    ax.plot(x_mirr, y, lw=2, label="fiber 2 mirrored")

    # physical fiber outer boundaries, perpendicular offset for accurate thickness
    def normal_vector(y, dxdy_fun):
        dxdy_vals = dxdy_fun(y)
        norm_length = np.sqrt(1.0 + dxdy_vals**2)
        # For curve (x(y), y): tangent is (dx/dy, 1), normal is (-1, dx/dy) rotated 90°
        return np.array([-np.ones_like(dxdy_vals), dxdy_vals]) / norm_length

    rf = 0.5 * cfg.diameter
    normal = normal_vector(y, dxdy_fun)
    normal_mirr = normal_vector(y, dxdy_mirr_fun)
    ax.plot(x - rf * normal[0], y - rf * normal[1], color="C0", alpha=0.25)
    ax.plot(x + rf * normal[0], y + rf * normal[1], color="C0", alpha=0.25)
    ax.plot(x_mirr - rf * normal_mirr[0], y - rf * normal_mirr[1], color="C1", alpha=0.25)
    ax.plot(x_mirr + rf * normal_mirr[0], y + rf * normal_mirr[1], color="C1", alpha=0.25)

    ax.axhline(info["y_front_end"], ls="--", alpha=0.5, color="gray")
    ax.axhline(info["y_middle_end"], ls="--", alpha=0.5, color="gray")

    if res is not None:
        y_contact = float(res)
        r_eff = peg_radius + 0.5 * cfg.diameter + clearance
        cx, cy = peg_center(x_fun, dxdy_fun, y_contact, r_eff, side_sign=1)
        peg = plt.Circle((cx, cy), peg_radius, fill=False, color="red", lw=2, label="peg")
        ax.add_patch(peg)
        ax.scatter([cx], [cy], color="red", zorder=5)
    else:
        print("No feasible constrained peg found in the middle interval.")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Static mirrored geometry with highest feasible peg")
    ax.grid(True)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    cfg = StaticCurveConfig(
        total_length=1.0,
        displacement=0.2,
        len_front=0.2,
        len_end=0.2,
    )
    
    plot_geometry_with_constraint_point(cfg, peg_radius=0.03, clearance=0.0, n=2000)