from itertools import combinations_with_replacement
from dataclasses import dataclass, replace
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq, minimize, Bounds


# ============================================================
# Configuration
# ============================================================

@dataclass
class StaticFiberConfig:
    total_length: float = 1.0
    displacement: float = 0.17
    diameter: float = 0.01
    len_front: float = 0.2
    len_end: float = 0.2

    # Inner curve model
    n_basis: int = 6
    n_quad: int = 121
    n_contact_sample: int = 181
    n_report: int = 401

    # Outer search
    outer_n_scan: int = 9
    outer_keep_top: int = 20

    # Solver
    inner_maxiter: int = 300
    smooth_alpha: float = 3000.0
    coeff_bound_factor: float = 1.5

    # Pegs
    peg_radius_default: float = 0.03
    clearance: float = 0.0

    # Acceptance tolerances
    length_tol: float = 5e-6
    penetration_tol: float = 5e-4
    contact_tol: float = 7e-4


# ============================================================
# Reference geometry
# ============================================================

def s(u):
    return 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3


def ds(u):
    return 30.0 * u**4 - 60.0 * u**3 + 30.0 * u**2


def d2s(u):
    return 120.0 * u**3 - 180.0 * u**2 + 60.0 * u


def middle_arc_length(Hm, d):
    return quad(
        lambda u: np.sqrt((d * ds(u))**2 + Hm**2),
        0.0,
        1.0,
        limit=2000,
    )[0]


def middle_height(cfg: StaticFiberConfig, Lf, Le):
    Lm = cfg.total_length - Lf - Le
    if Lm <= 0.0:
        raise ValueError("len_front + len_end must be smaller than total_length")

    def residual(H):
        return middle_arc_length(H, cfg.displacement) - Lm

    Hm = brentq(residual, 1e-12, Lm)
    return Hm, Lm


def build_reference_curve(cfg: StaticFiberConfig, Lf, Le):
    d = cfg.displacement
    Hm, Lm = middle_height(cfg, Lf, Le)

    y0 = 0.0
    y1 = Lf
    y2 = Lf + Hm
    y3 = Lf + Hm + Le

    def eta(y):
        return (y - y1) / Hm

    def x(y):
        y = np.asarray(y, dtype=np.float64)
        out = np.empty_like(y)

        mask_front = y < y1
        mask_mid = (y >= y1) & (y <= y2)
        mask_end = y > y2

        out[mask_front] = 0.0
        out[mask_mid] = d * s(eta(y[mask_mid]))
        out[mask_end] = d
        return out

    def dxdy(y):
        y = np.asarray(y, dtype=np.float64)
        out = np.zeros_like(y)

        mask_mid = (y >= y1) & (y <= y2)
        out[mask_mid] = d * ds(eta(y[mask_mid])) / Hm
        return out

    info = {
        "y_start": y0,
        "y_front_end": y1,
        "y_middle_end": y2,
        "y_end": y3,
        "middle_height": Hm,
        "middle_arc_length": Lm,
        "front_length": Lf,
        "end_length": Le,
    }

    return x, dxdy, info


def mirror_reference_curve(cfg: StaticFiberConfig, Lf, Le):
    x_ref, dxdy_ref, info = build_reference_curve(cfg, Lf, Le)

    def x_mirror(y):
        return cfg.diameter + 2.0 * cfg.displacement - x_ref(y)

    def dxdy_mirror(y):
        return -dxdy_ref(y)

    return x_mirror, dxdy_mirror, info


# ============================================================
# Peg placement on reference curve
# ============================================================

def peg_center_on_reference(x_fun, dxdy_fun, y_contact, r_eff, side_sign):
    slope = float(dxdy_fun(np.array([y_contact]))[0])
    norm = np.sqrt(1.0 + slope**2)
    normal = np.array([1.0, -slope]) / norm
    p_contact = np.array([float(x_fun(np.array([y_contact]))[0]), y_contact])
    return p_contact + side_sign * r_eff * normal


def build_pegs_from_reference(
    cfg,
    Lf,
    Le,
    y_contacts,
    side_signs,
    radii,
    clearance=None,
):
    if clearance is None:
        clearance = cfg.clearance

    x_ref, dxdy_ref, info = build_reference_curve(cfg, Lf, Le)

    y_contacts = np.asarray(y_contacts, dtype=np.float64)
    side_signs = np.asarray(side_signs, dtype=np.int64)
    radii = np.asarray(radii, dtype=np.float64)

    if len(y_contacts) != len(side_signs) or len(y_contacts) != len(radii):
        raise ValueError("y_contacts, side_signs, radii must have same length")

    if not np.all(np.diff(y_contacts) <= 0.0):
        raise ValueError("y_contacts must be ordered from top to bottom")

    centers = np.empty((len(y_contacts), 2), dtype=np.float64)

    y_min = info["y_front_end"]
    y_max = info["y_middle_end"]
    prev_y = y_max

    for i, (yc, side, r) in enumerate(zip(y_contacts, side_signs, radii)):
        r_eff = r + 0.5 * cfg.diameter + clearance
        if not (y_min <= yc <= prev_y):
            raise ValueError(f"Peg {i} has invalid y_contact={yc}")
        centers[i] = peg_center_on_reference(
            x_fun=x_ref,
            dxdy_fun=dxdy_ref,
            y_contact=yc,
            r_eff=r_eff,
            side_sign=int(side),
        )
        prev_y = yc

    return centers, info, x_ref


# ============================================================
# Basis functions for x(t)
#
# x(t) = d*s(t) + sum_k c_k * phi_k(t)
# y(t) = y_end * t
#
# This automatically enforces:
# x(0)=0, x(1)=d, x_t(0)=x_t(1)=0
# ============================================================

def basis_matrix(t, n_basis):
    t = np.asarray(t, dtype=np.float64)
    v = 2.0 * t - 1.0
    env = t**2 * (1.0 - t)**2
    mats = [env * v**k for k in range(n_basis)]
    return np.column_stack(mats)


def basis_d1_matrix(t, n_basis):
    t = np.asarray(t, dtype=np.float64)
    v = 2.0 * t - 1.0
    env = t**2 * (1.0 - t)**2
    denv = 2.0 * t * (1.0 - t) * (1.0 - 2.0 * t)

    mats = []
    for k in range(n_basis):
        term = denv * v**k
        if k > 0:
            term += env * k * v**(k - 1) * 2.0
        mats.append(term)
    return np.column_stack(mats)


def basis_d2_matrix(t, n_basis):
    t = np.asarray(t, dtype=np.float64)
    v = 2.0 * t - 1.0
    env = t**2 * (1.0 - t)**2
    denv = 2.0 * t * (1.0 - t) * (1.0 - 2.0 * t)
    d2env = 2.0 - 12.0 * t + 12.0 * t**2

    mats = []
    for k in range(n_basis):
        term = d2env * v**k
        if k > 0:
            term += 4.0 * denv * k * v**(k - 1)
        if k > 1:
            term += 4.0 * env * k * (k - 1) * v**(k - 2)
        mats.append(term)
    return np.column_stack(mats)


# ============================================================
# Elastic rod model
# ============================================================

class ElasticFiber2DModel:
    def __init__(self, cfg: StaticFiberConfig):
        self.cfg = cfg

        self.tq = np.linspace(0.0, 1.0, cfg.n_quad)
        self.tc = np.linspace(0.0, 1.0, cfg.n_contact_sample)
        self.tr = np.linspace(0.0, 1.0, cfg.n_report)

        self.P_q = basis_matrix(self.tq, cfg.n_basis)
        self.Pt_q = basis_d1_matrix(self.tq, cfg.n_basis)
        self.Ptt_q = basis_d2_matrix(self.tq, cfg.n_basis)

        self.P_c = basis_matrix(self.tc, cfg.n_basis)
        self.P_r = basis_matrix(self.tr, cfg.n_basis)

        self.base_q = s(self.tq)
        self.base_t_q = ds(self.tq)
        self.base_tt_q = d2s(self.tq)

        self.base_c = s(self.tc)
        self.base_r = s(self.tr)

    def unpack(self, z):
        c = np.asarray(z[:-1], dtype=np.float64)
        y_end = float(z[-1])
        return c, y_end

    def curve_on_quad_grid(self, z):
        c, y_end = self.unpack(z)
        x = self.cfg.displacement * self.base_q + self.P_q @ c
        xt = self.cfg.displacement * self.base_t_q + self.Pt_q @ c
        xtt = self.cfg.displacement * self.base_tt_q + self.Ptt_q @ c
        y = y_end * self.tq
        return x, y, xt, xtt, y_end

    def curve_on_contact_grid(self, z):
        c, y_end = self.unpack(z)
        x = self.cfg.displacement * self.base_c + self.P_c @ c
        y = y_end * self.tc
        return x, y

    def curve_on_report_grid(self, z):
        c, y_end = self.unpack(z)
        x = self.cfg.displacement * self.base_r + self.P_r @ c
        y = y_end * self.tr
        return x, y

    def length(self, z):
        _, _, xt, _, y_end = self.curve_on_quad_grid(z)
        speed = np.sqrt(xt**2 + y_end**2)
        return np.trapezoid(speed, self.tq)

    def bending_energy(self, z):
        _, _, xt, xtt, y_end = self.curve_on_quad_grid(z)
        denom = xt**2 + y_end**2
        integrand = (xtt**2 * y_end**2) / np.maximum(denom, 1e-14)**2.5
        return 0.5 * np.trapezoid(integrand, self.tq)

    def smooth_obstacle_constraints(self, z, peg_centers, r_eff_arr):
        x, y = self.curve_on_contact_grid(z)
        alpha = float(self.cfg.smooth_alpha)

        vals = np.empty(len(r_eff_arr), dtype=np.float64)
        for j, (pc, r_eff) in enumerate(zip(peg_centers, r_eff_arr)):
            v = (x - pc[0])**2 + (y - pc[1])**2 - r_eff**2
            m = np.min(v)
            vals[j] = m - np.log(np.sum(np.exp(-alpha * (v - m)))) / alpha
        return vals

    def true_contact_gaps(self, z, peg_centers, r_eff_arr):
        x, y = self.curve_on_report_grid(z)
        gaps = np.empty(len(r_eff_arr), dtype=np.float64)
        for j, (pc, r_eff) in enumerate(zip(peg_centers, r_eff_arr)):
            dist = np.sqrt((x - pc[0])**2 + (y - pc[1])**2)
            gaps[j] = np.min(dist) - r_eff
        return gaps

    def initial_guess_from_reference(self, x_ref=None, info_ref=None):
        if x_ref is None or info_ref is None:
            y_end0 = np.sqrt(max(self.cfg.total_length**2 - self.cfg.displacement**2, 1e-9))
            x_target = self.cfg.displacement * s(self.tq)
        else:
            y_end0 = float(info_ref["y_end"])
            x_target = x_ref(y_end0 * self.tq)

        rhs = x_target - self.cfg.displacement * self.base_q
        c0, *_ = np.linalg.lstsq(self.P_q, rhs, rcond=None)
        return np.concatenate([c0, [y_end0]])

    def project_guess_from_solution(self, sol):
        t = sol["t_dense"]
        x = sol["x_dense"]
        y_end = sol["y_end"]

        P = basis_matrix(t, self.cfg.n_basis)
        rhs = x - self.cfg.displacement * s(t)
        c0, *_ = np.linalg.lstsq(P, rhs, rcond=None)
        return np.concatenate([c0, [y_end]])


# ============================================================
# Inner solve
# ============================================================

def is_physically_acceptable(sol, cfg: StaticFiberConfig):
    return (
        abs(sol["length_error"]) <= cfg.length_tol
        and sol["max_penetration"] <= cfg.penetration_tol
    )


def solve_curve_for_fixed_pegs(
    cfg,
    peg_centers,
    peg_radii,
    clearance=None,
    x_ref=None,
    info_ref=None,
    z_init=None,
):
    if clearance is None:
        clearance = cfg.clearance

    model = ElasticFiber2DModel(cfg)
    r_eff_arr = np.asarray(peg_radii, dtype=np.float64) + 0.5 * cfg.diameter + float(clearance)

    if z_init is None:
        z0 = model.initial_guess_from_reference(x_ref=x_ref, info_ref=info_ref)
    else:
        z0 = np.asarray(z_init, dtype=np.float64).copy()

    coeff_bound = cfg.coeff_bound_factor * cfg.displacement
    lower = np.full(cfg.n_basis + 1, -coeff_bound, dtype=np.float64)
    upper = np.full(cfg.n_basis + 1, +coeff_bound, dtype=np.float64)

    lower[-1] = 1e-4
    upper[-1] = cfg.total_length

    bounds = Bounds(lower, upper)

    constraints = [
        {
            "type": "eq",
            "fun": lambda z: model.length(z) - cfg.total_length,
        },
        {
            "type": "ineq",
            "fun": lambda z: model.smooth_obstacle_constraints(z, peg_centers, r_eff_arr),
        },
    ]

    res = minimize(
        fun=model.bending_energy,
        x0=z0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": cfg.inner_maxiter,
            "ftol": 1e-10,
            "disp": False,
        },
    )

    z = np.asarray(res.x, dtype=np.float64)
    x_dense, y_dense = model.curve_on_report_grid(z)
    gaps = model.true_contact_gaps(z, peg_centers, r_eff_arr)
    length_now = model.length(z)
    length_error = length_now - cfg.total_length
    max_penetration = max(0.0, -float(np.min(gaps))) if len(gaps) else 0.0
    active_count = int(np.sum(gaps <= cfg.contact_tol))

    sol = {
        "optimizer_success": bool(res.success),
        "message": str(res.message),
        "status": int(res.status),
        "energy": float(model.bending_energy(z)),
        "length": float(length_now),
        "length_error": float(length_error),
        "x_dense": x_dense,
        "y_dense": y_dense,
        "t_dense": model.tr.copy(),
        "y_end": float(z[-1]),
        "z": z,
        "gaps": gaps,
        "active_count": active_count,
        "max_gap": float(np.max(gaps)) if len(gaps) else 0.0,
        "max_penetration": float(max_penetration),
        "accepted": False,
    }

    sol["accepted"] = is_physically_acceptable(sol, cfg)
    return sol


# ============================================================
# Candidate ranking
# ============================================================

def candidate_sort_key(result):
    sol = result["solution"]
    gaps = sol["gaps"]
    rms_gap = float(np.sqrt(np.mean(gaps**2))) if len(gaps) else 0.0

    # Prefer:
    # 1. more active pegs
    # 2. less penetration
    # 3. smaller overall gap
    # 4. lower bending energy
    return (
        -sol["active_count"],
        sol["max_penetration"],
        rms_gap,
        sol["energy"],
    )


# ============================================================
# Outer search
# ============================================================

def ordered_contact_tuples(y_grid, n_pegs):
    for idx_tuple in combinations_with_replacement(range(len(y_grid)), n_pegs):
        yield y_grid[np.array(idx_tuple[::-1], dtype=np.int64)]


def coarse_to_fine_search(cfg, side_signs, radii, n_scan=None, clearance=None):
    if n_scan is None:
        n_scan = cfg.outer_n_scan
    if clearance is None:
        clearance = cfg.clearance

    Lf = cfg.len_front
    Le = cfg.len_end

    _, _, info = build_reference_curve(cfg, Lf, Le)
    y_grid = np.linspace(info["y_front_end"], info["y_middle_end"], n_scan)
    n_pegs = len(side_signs)

    coarse_cfg = replace(
        cfg,
        n_basis=max(4, min(5, cfg.n_basis - 1)),
        n_quad=max(61, cfg.n_quad // 2),
        n_contact_sample=max(81, cfg.n_contact_sample // 2),
        n_report=max(201, cfg.n_report // 2),
        inner_maxiter=max(160, cfg.inner_maxiter // 2),
    )

    total_cases = math.comb(n_scan + n_pegs - 1, n_pegs)
    print(f"Scanning {total_cases} ordered peg layouts")

    coarse_hits = []
    case_counter = 0

    for y_contacts in ordered_contact_tuples(y_grid, n_pegs):
        case_counter += 1

        try:
            peg_centers, info_ref, x_ref = build_pegs_from_reference(
                cfg=coarse_cfg,
                Lf=Lf,
                Le=Le,
                y_contacts=y_contacts,
                side_signs=side_signs,
                radii=radii,
                clearance=clearance,
            )
        except ValueError:
            continue

        sol = solve_curve_for_fixed_pegs(
            cfg=coarse_cfg,
            peg_centers=peg_centers,
            peg_radii=radii,
            clearance=clearance,
            x_ref=x_ref,
            info_ref=info_ref,
            z_init=None,
        )

        if not sol["accepted"]:
            continue

        coarse_hits.append(
            {
                "y_contacts": y_contacts.copy(),
                "peg_centers": peg_centers.copy(),
                "solution": sol,
                "clearance": float(clearance),
            }
        )

        if case_counter % 50 == 0:
            print(
                f"  checked {case_counter}/{total_cases}, "
                f"accepted {len(coarse_hits)}"
            )

    if not coarse_hits:
        return []

    coarse_hits.sort(key=candidate_sort_key)
    coarse_hits = coarse_hits[: cfg.outer_keep_top]

    print(f"Refining top {len(coarse_hits)} coarse candidates")

    refined = []
    fine_model = ElasticFiber2DModel(cfg)

    for cand in coarse_hits:
        peg_centers, info_ref, x_ref = build_pegs_from_reference(
            cfg=cfg,
            Lf=Lf,
            Le=Le,
            y_contacts=cand["y_contacts"],
            side_signs=side_signs,
            radii=radii,
            clearance=clearance,
        )

        z0 = fine_model.project_guess_from_solution(cand["solution"])

        sol = solve_curve_for_fixed_pegs(
            cfg=cfg,
            peg_centers=peg_centers,
            peg_radii=radii,
            clearance=clearance,
            x_ref=x_ref,
            info_ref=info_ref,
            z_init=z0,
        )

        if not sol["accepted"]:
            continue

        refined.append(
            {
                "y_contacts": cand["y_contacts"].copy(),
                "peg_centers": peg_centers.copy(),
                "solution": sol,
                "clearance": float(clearance),
            }
        )

    refined.sort(key=candidate_sort_key)

    return refined if refined else coarse_hits


# ============================================================
# Plotting
# ============================================================

def plot_best_result(cfg, result, side_signs, radii):
    Lf = cfg.len_front
    Le = cfg.len_end

    x_ref, _, info = build_reference_curve(cfg, Lf, Le)
    x_mirror, _, _ = mirror_reference_curve(cfg, Lf, Le)

    y_ref = np.linspace(info["y_start"], info["y_end"], 2000)
    x_ref_vals = x_ref(y_ref)
    x_mirror_vals = x_mirror(y_ref)

    peg_centers = result["peg_centers"]
    sol = result["solution"]
    clearance_used = float(result.get("clearance", cfg.clearance))

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(x_ref_vals, y_ref, "--", alpha=0.45, label="reference curve")
    ax.plot(x_mirror_vals, y_ref, "--", alpha=0.25, label="mirrored reference")
    ax.plot(sol["x_dense"], sol["y_dense"], lw=2.5, label="elastic equilibrium")

    rf = 0.5 * cfg.diameter
    for c, r in zip(peg_centers, radii):
        peg = plt.Circle((c[0], c[1]), r, fill=False, color="red", lw=2)
        excl = plt.Circle(
            (c[0], c[1]),
            r + rf + clearance_used,
            fill=False,
            color="red",
            lw=1,
            ls="--",
            alpha=0.45,
        )
        ax.add_patch(peg)
        ax.add_patch(excl)
        ax.scatter([c[0]], [c[1]], color="red", zorder=5)

    title = (
        f"Best layout | active pegs = {sol['active_count']}/{len(radii)} | "
        f"energy = {sol['energy']:.6e} | "
        f"max penetration = {sol['max_penetration']:.3e}"
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()


def print_result_summary(result):
    sol = result["solution"]
    print("Accepted:", sol["accepted"])
    print("Optimizer success:", sol["optimizer_success"])
    print("Message:", sol["message"])
    print("Energy:", sol["energy"])
    print("Length error:", sol["length_error"])
    print("Active pegs:", sol["active_count"])
    print("Contact gaps:", sol["gaps"])
    print("Max penetration:", sol["max_penetration"])


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":
    cfg = StaticFiberConfig(
        total_length=1.0,
        displacement=0.17,
        diameter=0.001,
        len_front=0.2,
        len_end=0.2,
        n_basis=6,
        n_quad=121,
        n_contact_sample=181,
        n_report=401,
        outer_n_scan=9,
        outer_keep_top=20,
        inner_maxiter=300,
        smooth_alpha=3000.0,
        clearance=0.0,
        length_tol=5e-6,
        penetration_tol=5e-4,
        contact_tol=7e-4,
    )

    side_signs = [-1, +1, -1, +1]
    radii = [0.03, 0.03, 0.03, 0.03]

    results = coarse_to_fine_search(
        cfg=cfg,
        side_signs=side_signs,
        radii=radii,
        n_scan=cfg.outer_n_scan,
        clearance=cfg.clearance,
    )

    if not results:
        print("No physically acceptable peg layouts found.")
    else:
        best = results[0]
        print("Best y contacts:", best["y_contacts"])
        print("Best peg centers:\n", best["peg_centers"])
        print_result_summary(best)
        plot_best_result(cfg, best, side_signs, radii)