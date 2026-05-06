import numpy as  np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize, NonlinearConstraint
from numba import njit

from beam_elastica_plot import plot_solution
from numba_progress import ProgressBar



@njit(cache=True)
def ode_rhs(z, B):
    x, y, theta, M, H, V = z
    dz = np.empty(6, dtype=np.float64)
    dz[0] = np.cos(theta)
    dz[1] = np.sin(theta)
    dz[2] = M / B
    dz[3] = H * np.sin(theta) - V * np.cos(theta)
    dz[4] = 0.0
    dz[5] = 0.0
    return dz

def encode_interfaces(interfaces, total_length):
    m = len(interfaces)
    x_targets = np.full(m, np.nan, dtype=np.float64)
    y_targets = np.full(m, np.nan, dtype=np.float64)
    theta_targets = np.full(m, np.nan, dtype=np.float64)
    kind = np.full(m, -1, dtype=np.int64)  # 0 for load, 1 for constraint
    force_mag = np.zeros((m, 2), dtype=np.float64)
    normal_vec = np.zeros((m, 2), dtype=np.float64)
    has_normal = np.zeros(m, dtype=np.int64)

    for i, interface in enumerate(interfaces):
        
        if "axis" in interface:
            axis = interface["axis"]
            if axis == "x":
                x_value = interface.get("value", interface.get("x", np.nan))
                x_targets[i] = float(x_value)
            elif axis == "y":
                y_value = interface.get("value", interface.get("y", np.nan))
                y_targets[i] = float(y_value)
            else:
                raise ValueError("axis must be 'x' or 'y'.")
        else:
            if "x" in interface:
                x_targets[i] = float(interface["x"])
            if "y" in interface:
                y_targets[i] = float(interface["y"])
            if "theta" in interface:
                theta_targets[i] = float(interface["theta"])
        
        if interface.get("type") == "con":
            kind[i] = 1
        else:
            kind[i] = 0
            if "Py" in interface:
                force_mag[i, 0] = 1.0
                force_mag[i, 1] = float(interface["Py"])
            else:
                force_mag[i, 0] = 0.0
                force_mag[i, 1] = float(interface.get("force", 0.0))

        if "normal" in interface:
            n = np.asarray(interface["normal"], dtype=float)
            if n.shape != (2,):
                raise ValueError("normal must be a 2-vector [nx, ny].")
            n_norm = float(np.hypot(n[0], n[1]))
            if n_norm <= 0.0:
                raise ValueError("normal vector must be non-zero.")
            normal_vec[i, 0] = n[0] / n_norm
            normal_vec[i, 1] = n[1] / n_norm
            has_normal[i] = 1

    # Segment-length priors only seed the rho reparameterization.
    # If some interfaces are y-only (no x target), use a uniform prior.
    x_all_finite = np.all(np.isfinite(x_targets))
    if x_all_finite:
        xs = np.empty(m + 2, dtype=np.float64)
        xs[0] = 0.0
        xs[1:-1] = x_targets
        xs[-1] = float(total_length)
        dx = np.diff(xs)
        if np.all(dx > 0.0):
            ref_lengths = dx
        else:
            ref_lengths = np.full(m + 1, float(total_length) / (m + 1), dtype=np.float64)
    else:
        ref_lengths = np.full(m + 1, float(total_length) / (m + 1), dtype=np.float64)

    return x_targets, y_targets, theta_targets, kind, force_mag, normal_vec, has_normal, ref_lengths

def build_reference_lengths_from_targets(total_length, x_targets, y_targets, min_lengths):
    n_seg = len(min_lengths)
    available = float(total_length) - float(np.sum(min_lengths))
    if available <= 0.0:
        raise ValueError("total_length must exceed the sum of minimum segment lengths.")

    floor = max(1e-9 * float(total_length), 1e-9)
    ref_lengths = np.full(n_seg, available / n_seg, dtype=float)

    have_xy = np.all(np.isfinite(x_targets)) and np.all(np.isfinite(y_targets))
    if not have_xy or x_targets.size < 2:
        return ref_lengths

    points = np.column_stack((x_targets, y_targets))
    contact_chords = np.linalg.norm(np.diff(points, axis=0), axis=1)

    desired_lengths = np.array(min_lengths, dtype=float)
    desired_lengths[1:-1] += np.maximum(contact_chords, floor)

    spare = float(total_length) - float(np.sum(desired_lengths))
    if spare >= 0.0:
        desired_lengths[0] += 0.5 * spare
        desired_lengths[-1] += 0.5 * spare
    else:
        desired_lengths[0] += floor
        desired_lengths[-1] += floor

    return np.maximum(desired_lengths - min_lengths, floor)

def integrate_segment(z0, length, B):
    # Integrates the ODE and uses the initial conditions z0 to compute the state at the end of the segment. z(l) = z0 + integral of dz/ds from 0 to l.
    sol = solve_ivp(lambda s, z: ode_rhs(z, B), [0, length], z0, method='DOP853', rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError("ODE integration failed: " + sol.message)
    return sol.y[:, -1] 

def point_force(z_l, Px, Py):
    # Computes the change in the state due to a point force applied at the end of the segment. The point force changes the internal forces H and V, but not the position or angle.
    z_new = z_l.copy()
    z_new[4] -= Px
    z_new[5] -= Py
    return z_new


@njit(cache=True)
def lengths_segments(rho, total_length, ref_lengths, min_lengths):
    n = ref_lengths.size
    q = np.empty(n, dtype=np.float64)
    q[0] = ref_lengths[0]
    s = q[0]
    for i in range(1, n):
        q[i] = ref_lengths[i] * np.exp(rho[i - 1])
        s += q[i]

    remaining = total_length - np.sum(min_lengths)

    out = np.empty(n, dtype=np.float64)
    scale = remaining / s
    for i in range(n):
        out[i] = min_lengths[i] + q[i] * scale
    return out

@njit(cache=True)
def rk4_step(z, h, B):
    k1 = ode_rhs(z, B)
    k2 = ode_rhs(z + 0.5 * h * k1, B)
    k3 = ode_rhs(z + 0.5 * h * k2, B)
    k4 = ode_rhs(z + h * k3, B)
    return z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

@njit(cache=True)
def integrate_segment_rk4(z0, length, B, n_steps):
    z = z0.copy()
    if n_steps < 1:
        n_steps = 1
    h = length / n_steps
    for _ in range(n_steps):
        z = rk4_step(z, h, B)
    return z


@njit(cache=True)
def forward_march(u, total_length, B, x_targets, y_targets, kind, force_mag, normal_vec, has_normal, ref_lengths, min_lengths, n_steps):
    m = x_targets.size
    n_seg = m + 1
    n_rho = n_seg - 1

    x0 = u[0]
    y0 = u[1]
    theta0 = u[2]
    rho = u[3:3 + n_rho]
    lambdas = u[3 + n_rho:]

    lengths = lengths_segments(rho, total_length, ref_lengths, min_lengths)

    starts = np.empty((n_seg, 6), dtype=np.float64)
    ends = np.empty((n_seg, 6), dtype=np.float64)
    forces = np.zeros((m, 2), dtype=np.float64)

    z = np.array([x0, y0, theta0, 0.0, 0.0, 0.0], dtype=np.float64)
    lam_idx = 0

    for i in range(n_seg):
        starts[i] = z.copy()
        z_end = integrate_segment_rk4(z, lengths[i], B, n_steps)
        ends[i] = z_end

        if i < m:
            theta = z_end[2]
            if kind[i] == 1:
                lam = lambdas[lam_idx]
                if has_normal[i] > 0:
                    Px = lam * normal_vec[i, 0]
                    Py = lam * normal_vec[i, 1]
                else:
                    Px = -lam * np.sin(theta)
                    Py =  lam * np.cos(theta)
                lam_idx += 1
            else:
                if has_normal[i] > 0:
                    F = force_mag[i, 1]
                    Px = F * normal_vec[i, 0]
                    Py = F * normal_vec[i, 1]
                elif force_mag[i, 0] > 0.5:
                    Py = force_mag[i, 1]
                    Px = -Py * np.tan(theta)
                else:
                    F = force_mag[i, 1]
                    Px = -F * np.sin(theta)
                    Py =  F * np.cos(theta)

            forces[i, 0] = Px
            forces[i, 1] = Py

            z = z_end.copy()
            z[4] -= Px
            z[5] -= Py

    return starts, ends, forces, lengths

def reference_lengths(total_length, interfaces):
    xs = [0.0] + [float(interface["x"]) for interface in interfaces] + [float(total_length)]
    xs = np.asarray(xs, dtype=float)
    if np.any(np.diff(xs) <= 0.0):
        raise ValueError("x positions must be strictly increasing inside [0, total_length].")
    return np.diff(xs)



def make_residual(total_length, B, interfaces, n_steps=120, end_margin=0.0):
    x_targets, y_targets, theta_targets, kind, force_mag, normal_vec, has_normal, _ = encode_interfaces(interfaces, total_length)
    has_x = np.isfinite(x_targets)
    has_y = np.isfinite(y_targets)
    has_theta = np.isfinite(theta_targets)

    end_margin = float(end_margin)
    if end_margin < 0.0:
        raise ValueError("end_margin must be non-negative.")

    n_seg = len(interfaces) + 1
    min_lengths = build_min_lengths(total_length, n_seg, end_margin)
    ref_lengths = build_reference_lengths_from_targets(total_length, x_targets, y_targets, min_lengths)

    n_interface_eq = 0
    for i in range(len(interfaces)):
        n_targets = int(has_x[i]) + int(has_y[i]) + int(has_theta[i])
        if n_targets == 0:
            raise ValueError("Each interface must constrain at least one of x, y, or theta.")
        if kind[i] == 1:
            n_interface_eq += n_targets
        else:
            n_interface_eq += 1

    n_eq = n_interface_eq + 3
    target_specs = (
        (has_x, x_targets, 0),
        (has_y, y_targets, 1),
        (has_theta, theta_targets, 2),
    )

    def residuals(u):
        starts, ends, forces, lengths = forward_march(
            u, total_length, B,
            x_targets, y_targets, kind, force_mag, normal_vec, has_normal, ref_lengths, min_lengths,
            n_steps,
        )

        R = np.empty(n_eq, dtype=float)
        k = 0
        for i in range(len(interfaces)):
            if kind[i] == 1:
                for has_target, targets, col in target_specs:
                    if has_target[i]:
                        R[k] = ends[i, col] - targets[i]
                        k += 1
            else:
                for has_target, targets, col in target_specs:
                    if has_target[i]:
                        R[k] = ends[i, col] - targets[i]
                        k += 1
                        break

        R[k:k+3] = ends[-1, 3:6]
        return R

    return residuals



def count_unknowns(interfaces):
    return sum(1 for interface in interfaces if interface.get("type") == "con")

def unpack_reduced_unknowns(u, interfaces):
    n_interfaces = len(interfaces)
    n_segments = n_interfaces + 1
    n_rho = n_segments - 1
    n_lambda = count_unknowns(interfaces)

    x0 = u[0]
    y0 = u[1]
    theta0 = u[2]

    rho = np.asarray(u[3:3 + n_rho], dtype=float)
    lambdas = np.asarray(u[3 + n_rho:3 + n_rho + n_lambda], dtype=float)

    return x0, y0, theta0, rho, lambdas


@njit(cache=True)
def sample_segment(z0, length, B, points=200):
    n = int(points)
    if n < 2:
        n = 2

    out = np.empty((6, n), dtype=np.float64)
    z = z0.copy()
    out[:, 0] = z

    h = length / (n - 1)
    for i in range(1, n):
        z = rk4_step(z, h, B)
        out[:, i] = z

    return out

def beam_profile(solution, total_length, interfaces, B, points_per_segment=200, end_margin=0.0):
    x_targets, y_targets, theta_targets, kind, force_mag, normal_vec, has_normal, _ = encode_interfaces(interfaces, total_length)
    min_lengths = build_min_lengths(total_length, len(interfaces) + 1, end_margin)
    ref_lengths = build_reference_lengths_from_targets(total_length, x_targets, y_targets, min_lengths)

    starts, ends, forces, lengths = forward_march(solution, total_length, B,
            x_targets, y_targets, kind, force_mag, normal_vec, has_normal, ref_lengths, min_lengths, 400)

    sampled_segments = [
        sample_segment(starts[i], lengths[i], B, points=points_per_segment)
        for i in range(len(lengths))
    ]

    x = np.concatenate([seg[0] for seg in sampled_segments])
    y = np.concatenate([seg[1] for seg in sampled_segments])
    theta = np.concatenate([seg[2] for seg in sampled_segments])

    joints = np.array([[ends[i][0], ends[i][1]] for i in range(len(interfaces))], dtype=float)

    return x, y, theta, joints, forces

def build_min_lengths(total_length, n_seg, end_margin=0.0):
    end_margin = float(end_margin)
    if end_margin < 0.0:
        raise ValueError("end_margin must be non-negative.")
    if n_seg >= 2 and 2.0 * end_margin >= float(total_length):
        raise ValueError("require 2 * end_margin < total_length")

    min_lengths = np.zeros(n_seg, dtype=float)
    if n_seg >= 2 and end_margin > 0.0:
        min_lengths[0] = end_margin
        min_lengths[-1] = end_margin
    return min_lengths




def bending_energy(solution, total_length, B, interfaces, points_per_segment=200, end_margin=0.0, n_steps=400):
    x_targ, y_targ, theta_targ, kind, force_mag, normal_vec, has_normal, _ = encode_interfaces(interfaces, total_length)
    min_lengths = build_min_lengths(total_length, len(interfaces) + 1, end_margin)
    ref_lengths = build_reference_lengths_from_targets(total_length, x_targ, y_targ, min_lengths)
    starts, ends, forces, lengths = forward_march(solution, total_length, B, x_targ, y_targ, kind, force_mag, normal_vec, has_normal, ref_lengths, min_lengths, n_steps)
    
    energy = 0.0
    for i in range(len(lengths)):
        seg = sample_segment(starts[i], lengths[i], B, points=points_per_segment)
        M = seg[3]
        ds = lengths[i] / (M.size - 1)
        energy +=  np.sum(M**2) * ds
    return 0.5 * energy / B



def initial_guess(interfaces, total_length=None, end_margin=0.0):
    n_segments = len(interfaces) + 1
    n_rho = n_segments - 1
    n_lambda = count_unknowns(interfaces)

    x0 = 0.0
    y0 = 0.0
    theta0 = 0.0
    rho = np.zeros(n_rho, dtype=float)
    lambdas = np.zeros(n_lambda, dtype=float)

    if total_length is not None and len(interfaces) > 0:
        x_targets, y_targets, theta_targets, kind, force_mag, normal_vec, has_normal, _ = encode_interfaces(interfaces, total_length)
        min_lengths = build_min_lengths(total_length, n_segments, end_margin)
        ref_lengths = build_reference_lengths_from_targets(total_length, x_targets, y_targets, min_lengths)
        lengths = lengths_segments(rho, total_length, ref_lengths, min_lengths)

        if np.isfinite(x_targets[0]) and np.isfinite(y_targets[0]):
            if np.isfinite(theta_targets[0]):
                theta0 = theta_targets[0]
            else:
                theta0 = 0.0
            x0 = x_targets[0] - lengths[0] * np.cos(theta0)
            y0 = y_targets[0] - lengths[0] * np.sin(theta0)

    return np.concatenate([[x0, y0, theta0], rho, lambdas])

def build_unknown_bounds(interfaces):
    n_segments = len(interfaces) + 1
    n_rho = n_segments - 1
    n_lambda = count_unknowns(interfaces)

    n_total = 3 + n_rho + n_lambda
    lb = np.full(n_total, -np.inf, dtype=float)
    ub = np.full(n_total, np.inf, dtype=float)

    lam_idx = 3 + n_rho
    for interface in interfaces:
        if interface.get("type") == "con":
            if "normal" in interface:
                lb[lam_idx] = 0.0
            lam_idx += 1

    return lb, ub

def solve_segments(total_length, B, interfaces, u0=None, theta0=None, end_margin=0.0, max_nfev=1000, n_steps=400):
    u = initial_guess(interfaces, total_length=total_length, end_margin=end_margin) if u0 is None else np.asarray(u0, dtype=float)
    if theta0 is not None:
        u[2] = float(theta0)

    base_resfun = make_residual(total_length, B, interfaces, n_steps=n_steps, end_margin=end_margin)
    if theta0 is not None:
        theta0_value = float(theta0)

        def resfun(u_vec):
            u_eval = u_vec.copy()
            u_eval[2] = theta0_value
            return base_resfun(u_eval)
    else:
        resfun = base_resfun

    lb, ub = build_unknown_bounds(interfaces)
    if theta0 is not None:
        theta0_value = float(theta0)
        eps = np.sqrt(np.finfo(float).eps)
        lb[2] = theta0_value - eps
        ub[2] = theta0_value + eps

    bounds = (lb, ub)
    finite_lb = np.isfinite(lb)
    finite_ub = np.isfinite(ub)
    u[finite_lb] = np.maximum(u[finite_lb], lb[finite_lb])
    u[finite_ub] = np.minimum(u[finite_ub], ub[finite_ub])

    res = least_squares(
        resfun,
        u,
        bounds=bounds,
        jac='2-point',
        method='trf',
        x_scale='jac',
        xtol=1e-8,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=max_nfev,
        verbose=0,
    )
    u = res.x
    return res

def solution_metrics(solution, total_length, B, interfaces, theta0=None, end_margin=0.0, n_steps=400):
    solution_eval = np.asarray(solution, dtype=float)
    if theta0 is not None:
        solution_eval = solution_eval.copy()
        solution_eval[2] = float(theta0)

    x_targ, y_targ, theta_targ, kind, force_mag, normal_vec, has_normal, _ = encode_interfaces(interfaces, total_length)
    min_lengths = build_min_lengths(total_length, len(interfaces) + 1, end_margin)
    ref_lengths = build_reference_lengths_from_targets(total_length, x_targ, y_targ, min_lengths)
    starts, ends, forces, lengths = forward_march(
        solution_eval,
        total_length,
        B,
        x_targ,
        y_targ,
        kind,
        force_mag,
        normal_vec,
        has_normal,
        ref_lengths,
        min_lengths,
        n_steps,
    )

    return starts, ends, forces, lengths


def physical_diagnostics(solution, total_length, B, interfaces, theta0=None, end_margin=0.0, n_steps=400):
    solution_eval = np.asarray(solution, dtype=float)
    if theta0 is not None:
        solution_eval = solution_eval.copy()
        solution_eval[2] = float(theta0)

    r = make_residual(total_length, B, interfaces, end_margin=end_margin, n_steps=n_steps)(solution_eval)
    starts, ends, forces, lengths = solution_metrics(
        solution_eval,
        total_length,
        B,
        interfaces,
        theta0=None,
        end_margin=end_margin,
        n_steps=n_steps,
    )

    left_tail_delta = ends[0, 2] - starts[0, 2]
    right_tail_delta = ends[-1, 2] - starts[-1, 2]
    tail_straightness = max(abs(left_tail_delta), abs(right_tail_delta))
    tail_state_max = max(
        np.max(np.abs(starts[0, 3:6])),
        np.max(np.abs(starts[-1, 3:6])),
        np.max(np.abs(ends[-1, 3:6])),
    )

    return {
        "residual_norm": float(np.linalg.norm(r)),
        "max_abs_residual": float(np.max(np.abs(r))),
        "left_angle": float(starts[0, 2]),
        "right_contact_angle": float(starts[-1, 2]),
        "right_end_angle": float(ends[-1, 2]),
        "left_tail_delta": float(left_tail_delta),
        "right_tail_delta": float(right_tail_delta),
        "tail_straightness": float(tail_straightness),
        "tail_state_max": float(tail_state_max),
        "lengths": lengths,
        "forces": forces,
    }


def is_physical_solution(solution, total_length, B, interfaces, theta0=None, end_margin=0.0, n_steps=400, residual_tol=1e-6, tail_angle_tol=np.deg2rad(0.05)):
    diag = physical_diagnostics(
        solution,
        total_length,
        B,
        interfaces,
        theta0=theta0,
        end_margin=end_margin,
        n_steps=n_steps,
    )
    values = [
        diag["residual_norm"],
        diag["max_abs_residual"],
        diag["left_angle"],
        diag["right_contact_angle"],
        diag["right_end_angle"],
        diag["tail_straightness"],
    ]
    return (
        np.all(np.isfinite(values))
        and diag["max_abs_residual"] <= residual_tol
        and diag["tail_straightness"] <= tail_angle_tol
    )


def continuity_diagnostics(solution, total_length, B, interfaces, theta0=None, end_margin=0.0, n_steps=400):
    diag = physical_diagnostics(
        solution,
        total_length,
        B,
        interfaces,
        theta0=theta0,
        end_margin=end_margin,
        n_steps=n_steps,
    )
    print(f"max abs residual = {diag['max_abs_residual']:.4g}")
    print(f"residual norm    = {diag['residual_norm']:.4g}")
    print(
        "tail angles deg  = "
        f"left {diag['left_angle'] * 180 / np.pi:.4g}, "
        f"right contact {diag['right_contact_angle'] * 180 / np.pi:.4g}, "
        f"right end {diag['right_end_angle'] * 180 / np.pi:.4g}"
    )
    print(
        "tail delta deg   = "
        f"left {diag['left_tail_delta'] * 180 / np.pi:.4g}, "
        f"right {diag['right_tail_delta'] * 180 / np.pi:.4g}"
    )
    print(f"tail state max   = {diag['tail_state_max']:.4g}")
    print(f"segment lengths  = {np.array2string(diag['lengths'], precision=4)}")



def solve_inverse_guess(d, total_length, B, interfaces, u0=None, theta0=None, end_margin=0.0, max_nfev=300, n_steps=400):
    interfaces = [dict(interface) for interface in interfaces]
    for i in range(1, len(interfaces)-1):
        interfaces[i]["x"] = d[2*i - 2]
        interfaces[i]["y"] = d[2*i - 1]

    res = solve_segments(total_length, B, interfaces, u0=u0, theta0=theta0, end_margin=end_margin, max_nfev=max_nfev, n_steps=n_steps)
    return res, interfaces

def solve_multistart(d, total_length, B, interfaces, seeds, theta0=None, end_margin=0.0, max_nfev=300, n_steps=400):
    seeds = list(seeds)
    trial_interfaces = [dict(interface) for interface in interfaces]
    for i in range(1, len(trial_interfaces)-1):
        trial_interfaces[i]["x"] = d[2*i - 2]
        trial_interfaces[i]["y"] = d[2*i - 1]

    n_rho = len(trial_interfaces)
    n_lambda = count_unknowns(trial_interfaces)
    lam_start = 3 + n_rho
    if n_lambda > 0:
        base_seed = initial_guess(trial_interfaces, total_length=total_length, end_margin=end_margin)
        force_seed = max(1e-6, 0.1 * abs(float(B)))
        patterns = [
            np.ones(n_lambda, dtype=float),
            -np.ones(n_lambda, dtype=float),
            np.where(np.arange(n_lambda) % 2 == 0, 1.0, -1.0),
            np.where(np.arange(n_lambda) % 2 == 0, -1.0, 1.0),
        ]
        for pattern in patterns:
            seed = base_seed.copy()
            seed[lam_start:lam_start + n_lambda] = force_seed * pattern
            seeds.append(seed)

    best = None
    best_interfaces = None
    best_score = np.inf
    last = None
    last_interfaces = None
    for u0 in seeds:
        res, trial_interfaces = solve_inverse_guess(
            d,
            total_length,
            B,
            interfaces,
            u0=u0,
            theta0=theta0,
            end_margin=end_margin,
            max_nfev=max_nfev,
            n_steps=n_steps,
        )
        last = res
        last_interfaces = trial_interfaces
        diag = physical_diagnostics(
            res.x,
            total_length,
            B,
            trial_interfaces,
            theta0=theta0,
            end_margin=end_margin,
            n_steps=n_steps,
        )
        eq_penalty = np.dot(res.fun, res.fun)
        score = eq_penalty + diag["tail_straightness"]**2 + diag["tail_state_max"]**2
        if np.isfinite(score) and score < best_score:
            best = res
            best_interfaces = trial_interfaces
            best_score = score

    if best is None:
        return last, last_interfaces

    return best, best_interfaces


def design_points_from_vector(d, interfaces):
    points = [(interfaces[0]["x"], interfaces[0]["y"])]
    for i in range(1, len(interfaces) - 1):
        points.append((d[2*i - 2], d[2*i - 1]))
    points.append((interfaces[-1]["x"], interfaces[-1]["y"]))
    return np.asarray(points, dtype=float)


def contact_chord_slack(d, interfaces, total_length, end_margin=0.0):
    points = design_points_from_vector(d, interfaces)
    chord_sum = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    min_lengths = build_min_lengths(total_length, len(interfaces) + 1, end_margin)
    available_contact_length = float(total_length) - min_lengths[0] - min_lengths[-1]
    return available_contact_length - chord_sum


def design_vector_from_interfaces(interfaces):
    if len(interfaces) <= 2:
        return np.empty(0, dtype=float)
    return np.concatenate([
        np.array([interface["x"], interface["y"]], dtype=float)
        for interface in interfaces[1:-1]
    ])


def apply_design_vector(d, interfaces):
    moved = [dict(interface) for interface in interfaces]
    for i in range(1, len(moved) - 1):
        moved[i]["x"] = float(d[2*i - 2])
        moved[i]["y"] = float(d[2*i - 1])
    return moved


def build_design_bounds(interfaces, radius=2.0, margin=0.1):
    bounds = []
    for interface in interfaces[1:-1]:
        for axis in ("x", "y"):
            lower = float(interface[axis]) - radius
            upper = float(interface[axis]) + radius
            end_upper = float(interfaces[-1][axis]) - margin
            if upper > end_upper:
                upper = end_upper
            bounds.append((lower, upper))
    return bounds


def design_guard_residuals(d, interfaces, total_length, end_margin=0.0, min_dx=0.1):
    if len(interfaces) <= 2:
        return np.empty(0, dtype=float)

    guards = []
    xs = [float(interfaces[0]["x"])]
    xs.extend(float(d[2*i]) for i in range(len(interfaces) - 2))
    xs.append(float(interfaces[-1]["x"]))

    for left, right in zip(xs[:-1], xs[1:]):
        guards.append(min(0.0, right - left - min_dx))
    guards.append(min(0.0, contact_chord_slack(d, interfaces, total_length, end_margin)))

    return np.asarray(guards, dtype=float)


def solve_coupled_design(
    total_length,
    B,
    base_interfaces,
    theta0=None,
    end_margin=0.0,
    force_index=0,
    force_weights=(0.0, 1.0, 100.0),
    radius=2.0,
    margin=0.1,
    min_dx=0.1,
    displacement_weight=1e-3,
    guard_weight=100.0,
    n_steps=120,
    max_nfev=3000,
    polish_n_steps=400,
    polish_max_nfev=4000,
    residual_tol=1e-5,
    tail_angle_tol=np.deg2rad(0.05),
    show_progress=True,
):
    if not -len(base_interfaces) <= force_index < len(base_interfaces):
        raise ValueError("force_index must select one interface.")

    d0 = design_vector_from_interfaces(base_interfaces)
    design_bounds = build_design_bounds(base_interfaces, radius=radius, margin=margin)

    u0 = initial_guess(base_interfaces, total_length=total_length, end_margin=end_margin)
    if theta0 is not None:
        u0[2] = float(theta0)

    nd = d0.size
    w0 = np.concatenate([d0, u0])
    lb = np.full(w0.size, -np.inf, dtype=float)
    ub = np.full(w0.size, np.inf, dtype=float)

    for i, (lower, upper) in enumerate(design_bounds):
        lb[i] = lower
        ub[i] = upper

    u_lb, u_ub = build_unknown_bounds(base_interfaces)
    if theta0 is not None:
        theta0_value = float(theta0)
        eps = np.sqrt(np.finfo(float).eps)
        u_lb[2] = theta0_value - eps
        u_ub[2] = theta0_value + eps

    lb[nd:] = u_lb
    ub[nd:] = u_ub
    finite_lb = np.isfinite(lb)
    finite_ub = np.isfinite(ub)
    w0[finite_lb] = np.maximum(w0[finite_lb], lb[finite_lb])
    w0[finite_ub] = np.minimum(w0[finite_ub], ub[finite_ub])

    selected_force_index = force_index if force_index >= 0 else len(base_interfaces) + force_index
    history = []
    best = None
    best_score = np.inf

    for force_weight in force_weights:
        sqrt_force_weight = np.sqrt(float(force_weight))

        def residuals(w):
            d = w[:nd]
            u = w[nd:].copy()
            if theta0 is not None:
                u[2] = float(theta0)

            interfaces = apply_design_vector(d, base_interfaces)
            model_residual = make_residual(
                total_length,
                B,
                interfaces,
                n_steps=n_steps,
                end_margin=end_margin,
            )(u)

            parts = [
                model_residual,
                displacement_weight * (d - d0),
                guard_weight * design_guard_residuals(
                    d,
                    interfaces,
                    total_length,
                    end_margin=end_margin,
                    min_dx=min_dx,
                ),
            ]

            if sqrt_force_weight > 0.0:
                starts, ends, forces, lengths = solution_metrics(
                    u,
                    total_length,
                    B,
                    interfaces,
                    theta0=theta0,
                    end_margin=end_margin,
                    n_steps=n_steps,
                )
                parts.append(np.array([sqrt_force_weight * np.linalg.norm(forces[selected_force_index])]))

            return np.concatenate(parts)

        least_squares_kwargs = {
            "bounds": (lb, ub),
            "method": "trf",
            "x_scale": "jac",
            "ftol": 1e-10,
            "xtol": 1e-10,
            "gtol": 1e-10,
            "max_nfev": max_nfev,
            "verbose": 0,
        }
        if show_progress:
            with ProgressBar(total=max_nfev, desc=f"Coupled force={force_weight:g}", unit="step", leave=False) as pbar:
                def progress_callback(*args, **kwargs):
                    pbar.update()

                stage_res = least_squares(
                    residuals,
                    w0,
                    callback=progress_callback,
                    **least_squares_kwargs,
                )
        else:
            stage_res = least_squares(
                residuals,
                w0,
                **least_squares_kwargs,
            )

        w0 = stage_res.x.copy()
        d = w0[:nd]
        interfaces = apply_design_vector(d, base_interfaces)
        u = w0[nd:].copy()
        if theta0 is not None:
            u[2] = float(theta0)

        polished = solve_segments(
            total_length,
            B,
            interfaces,
            u0=u,
            theta0=theta0,
            end_margin=end_margin,
            max_nfev=polish_max_nfev,
            n_steps=polish_n_steps,
        )
        diag = physical_diagnostics(
            polished.x,
            total_length,
            B,
            interfaces,
            theta0=theta0,
            end_margin=end_margin,
            n_steps=polish_n_steps,
        )
        physical = is_physical_solution(
            polished.x,
            total_length,
            B,
            interfaces,
            theta0=theta0,
            end_margin=end_margin,
            n_steps=polish_n_steps,
            residual_tol=residual_tol,
            tail_angle_tol=tail_angle_tol,
        )
        target_force = float(np.linalg.norm(diag["forces"][selected_force_index]))
        score = target_force if physical else np.inf

        entry = {
            "force_weight": float(force_weight),
            "stage_result": stage_res,
            "solution": polished,
            "interfaces": interfaces,
            "design": d.copy(),
            "diagnostics": diag,
            "target_force": target_force,
            "physical": physical,
        }
        history.append(entry)

        if score < best_score:
            best = entry
            best_score = score

    if best is None:
        best = history[-1]

    return best["solution"], best["interfaces"], best["design"], history


def top_coupled_candidates(history, top_n=3, physical_only=True):
    candidates = []
    for entry in history:
        if physical_only and not entry["physical"]:
            continue
        candidates.append(entry)

    candidates.sort(key=lambda entry: entry["target_force"])
    return candidates[:top_n]


def print_coupled_candidate_leaderboard(history, top_n=3, physical_only=True):
    candidates = top_coupled_candidates(history, top_n=top_n, physical_only=physical_only)
    if not candidates:
        print("No physical candidates found for leaderboard.")
        return

    print(f"Top {len(candidates)} candidates by selected contact force:")
    for rank, entry in enumerate(candidates, start=1):
        diag = entry["diagnostics"]
        print(
            f"{rank}. force={entry['target_force']:.6g}, "
            f"force_weight={entry['force_weight']:.4g}, "
            f"max_res={diag['max_abs_residual']:.4g}, "
            f"tail_delta={diag['tail_straightness'] * 180 / np.pi:.4g} deg"
        )
        print(np.array2string(entry["design"], precision=6))


def remove_theta_targets(interfaces):
    free_interfaces = [dict(interface) for interface in interfaces]
    for interface in free_interfaces:
        interface.pop("theta", None)
    return free_interfaces


def solve_free_evaluation(total_length, B, interfaces, u0=None, end_margin=0.0, max_nfev=4000, n_steps=400):
    free_interfaces = remove_theta_targets(interfaces)
    res = solve_segments(
        total_length,
        B,
        free_interfaces,
        u0=u0,
        theta0=None,
        end_margin=end_margin,
        max_nfev=max_nfev,
        n_steps=n_steps,
    )
    return res, free_interfaces


def make_outer_objective(
    total_length,
    B,
    base_interfaces,
    cache=True,
    theta0=None,
    end_margin=0.0,
    max_nfev=80,
    n_steps=400,
    force_index=0,
    force_weight=1e4,
    eq_weight=1e6,
    energy_weight=1.0,
    horizontal_weight=1e4,
    tail_weight=1e8,
    residual_tol=1e-6,
    tail_angle_tol=np.deg2rad(0.05),
):
    if not -len(base_interfaces) <= force_index < len(base_interfaces):
        raise ValueError("force_index must select one interface.")

    cached = {"solution": None}
    def objective(d):
        seeds = [None]
        if cache and cached["solution"] is not None:
            seeds.append(cached["solution"])

        res, interfaces = solve_multistart(
            d,
            total_length,
            B,
            base_interfaces,
            seeds,
            theta0=theta0,
            end_margin=end_margin,
            max_nfev=max_nfev,
            n_steps=n_steps,
        )
        eq_penalty = np.dot(res.fun, res.fun)

        diag = physical_diagnostics(
            res.x,
            total_length,
            B,
            interfaces,
            theta0=theta0,
            end_margin=end_margin,
            n_steps=n_steps,
        )
        selected_force_index = force_index if force_index >= 0 else len(interfaces) + force_index
        target_force = np.linalg.norm(diag["forces"][selected_force_index])
        horizontal_penalty = (
            np.sin(diag["left_angle"])**2
            + np.sin(diag["right_contact_angle"])**2
            + np.sin(diag["right_end_angle"])**2
        )
        tail_penalty = diag["left_tail_delta"]**2 + diag["right_tail_delta"]**2

        E = bending_energy(res.x, total_length, B, interfaces, end_margin=end_margin, n_steps=n_steps)
        if not np.all(np.isfinite([eq_penalty, target_force, horizontal_penalty, tail_penalty, E])):
            return 1e30

        J = (
            eq_weight * eq_penalty
            + force_weight * target_force**2
            + energy_weight * E
            + horizontal_weight * horizontal_penalty
            + tail_weight * tail_penalty
        )

        residual_excess = max(0.0, diag["max_abs_residual"] - residual_tol)
        tail_excess = max(0.0, diag["tail_straightness"] - tail_angle_tol)
        if residual_excess > 0.0 or tail_excess > 0.0:
            J += 1e9 + 1e12 * (residual_excess**2 + tail_excess**2)
        elif cache:
            cached["solution"] = res.x.copy()

        return J
        
    return objective

def main():
    
    interfaces = [
        {"type": "con", "x": 0, "y": 0, "normal": [0.0, 1.0]},
        {"type": "con", "x": 2.388, "y": 1.037e-05},
        {"type": "con", "x": 6.29, "y": 2.238},
        {"type": "con", "x": 6.39, "y": 3.54},
        {"type": "con", "x": 8, "y": 4.1725, "normal": [0.0, -1.0]},
    ]

    total_length = 15.0
    end_margin = 0.5
    theta0 = None
    B = 8.5e-3

    sol = solve_segments(total_length, B, interfaces=interfaces, theta0=theta0, end_margin=end_margin, max_nfev=2000, n_steps=400)

    print("Initial free evaluation:")
    continuity_diagnostics(sol.x, total_length, B, interfaces, theta0, end_margin)
    plot_solution(
        sol.x,
        total_length,
        interfaces,
        B,
        beam_profile,
        solve_segments,
        beam_profile_kwargs={"end_margin": end_margin},
        solve_segments_kwargs={"theta0": theta0, "end_margin": end_margin},
    )

    theta0 = 0.0
    interfaces[-1]["theta"] = 0.0
    constraint = 0.1

    sol_check, interfaces_check, optimal_d, history = solve_coupled_design(
        total_length,
        B,
        interfaces,
        theta0=theta0,
        end_margin=end_margin,
        force_index=0,
        force_weights=(0.0, 1.0, 10.0, 100.0, 300.0),
        radius=2.0,
        margin=constraint,
        min_dx=constraint,
        residual_tol=1e-5,
    )

    for entry in history:
        diag = entry["diagnostics"]
        print(
            f"stage force_weight={entry['force_weight']:.4g}: "
            f"physical={entry['physical']}, "
            f"target_force={entry['target_force']:.4g}, "
            f"max_res={diag['max_abs_residual']:.4g}, "
            f"tail_delta={diag['tail_straightness'] * 180 / np.pi:.4g} deg"
        )

    print_coupled_candidate_leaderboard(history, top_n=3, physical_only=True)

    continuity_diagnostics(sol_check.x, total_length, B, interfaces_check, theta0=0.0, end_margin=end_margin)
    if is_physical_solution(sol_check.x, total_length, B, interfaces_check, theta0=0.0, end_margin=end_margin, residual_tol=1e-5):
        print(f"Optimal interface positions:")
        print(np.array2string(optimal_d, precision=6))
        print(f"inner cost = {sol_check.cost:.4g}")
        plot_solution(
            sol_check.x,
            total_length,
            interfaces_check,
            B,
            beam_profile,
            solve_segments,
            beam_profile_kwargs={"end_margin": end_margin},
            solve_segments_kwargs={"theta0": 0.0, "end_margin": end_margin},
        )
    else:
        print("Constrained design candidate did not pass the physical gate; continuing to free evaluation.")

    free_sol, free_interfaces = solve_free_evaluation(
        total_length,
        B,
        interfaces_check,
        u0=sol_check.x,
        end_margin=end_margin,
        max_nfev=4000,
        n_steps=400,
    )
    print("Final free evaluation with theta constraints removed:")
    continuity_diagnostics(free_sol.x, total_length, B, free_interfaces, theta0=None, end_margin=end_margin)
    plot_solution(
        free_sol.x,
        total_length,
        free_interfaces,
        B,
        beam_profile,
        solve_segments,
        beam_profile_kwargs={"end_margin": end_margin},
        solve_segments_kwargs={"theta0": None, "end_margin": end_margin},
    )


if __name__ == "__main__":
    main()
