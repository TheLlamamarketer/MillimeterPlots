import numpy as  np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize, NonlinearConstraint
from numba import njit

from beam_elastica_plot import plot_solution



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
    x_targets, y_targets, theta_targets, kind, force_mag, normal_vec, has_normal, ref_lengths = encode_interfaces(interfaces, total_length)
    has_x = np.isfinite(x_targets)
    has_y = np.isfinite(y_targets)
    has_theta = np.isfinite(theta_targets)

    end_margin = float(end_margin)
    if end_margin < 0.0:
        raise ValueError("end_margin must be non-negative.")

    n_seg = len(interfaces) + 1
    min_lengths = build_min_lengths(total_length, n_seg, end_margin)

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
    x_targets, y_targets, theta_targets, kind, force_mag, normal_vec, has_normal, ref_lengths = encode_interfaces(interfaces, total_length)
    min_lengths = build_min_lengths(total_length, len(interfaces) + 1, end_margin)

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




def bending_energy(solution, total_length, B, interfaces, points_per_segment=200, end_margin=0.0):
    x_targ, y_targ, theta_targ, kind, force_mag, normal_vec, has_normal, ref_lengths = encode_interfaces(interfaces, total_length)
    min_lengths = build_min_lengths(total_length, len(interfaces) + 1, end_margin)
    starts, ends, forces, lengths = forward_march(solution, total_length, B, x_targ, y_targ, kind, force_mag, normal_vec, has_normal, ref_lengths, min_lengths, n_steps=400)
    
    energy = 0.0
    for i in range(len(lengths)):
        seg = sample_segment(starts[i], lengths[i], B, points=points_per_segment)
        M = seg[3]
        ds = lengths[i] / (M.size - 1)
        energy +=  np.sum(M**2) * ds
    return 0.5 * energy / B



def initial_guess(total_length, interfaces):
    n_segments = len(interfaces) + 1
    n_rho = n_segments - 1
    n_lambda = count_unknowns(interfaces)

    x0 = 0.0
    y0 = 0.0
    theta0 = 0.0
    rho = np.zeros(n_rho, dtype=float)
    lambdas = np.zeros(n_lambda, dtype=float)

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

def solve_segments(total_length, B, interfaces, u0=None, theta0=None, end_margin=0.0, max_nfev=1000):
    u = initial_guess(total_length, interfaces) if u0 is None else np.asarray(u0, dtype=float)
    if theta0 is not None:
        u[2] = float(theta0)

    base_resfun = make_residual(total_length, B, interfaces, n_steps=400, end_margin=end_margin)
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
        verbose=1,
    )
    u = res.x
    return res

def continuity_diagnostics(solution, total_length, B, interfaces, theta0=None, end_margin=0.0):
    solution_eval = np.asarray(solution, dtype=float)
    if theta0 is not None:
        solution_eval = solution_eval.copy()
        solution_eval[2] = float(theta0)

    r = make_residual(total_length, B, interfaces, end_margin=end_margin)(solution_eval)
    print("max abs residual =", np.max(np.abs(r)))
    print("residual norm    =", np.linalg.norm(r))


interfaces = [
    {"type": "con", "x": 0, "y": 0, "normal": [0.0, 1.0]},
    {"type": "con", "x": 1.575, "y": 0.425, "normal": [1.0, -1.0]},
    {"type": "con", "x": 2, "y": 2, "normal": [-1.0, 0.0], "theta": np.pi / 2},
]

total_length = 6.0
theta0 = 0.0
end_margin = 1.0




sol = solve_segments(total_length, 1.0, interfaces=interfaces, theta0=theta0, end_margin=end_margin)

continuity_diagnostics(sol.x, total_length, 1.0, interfaces, theta0, end_margin)

plot_solution(
    sol.x,
    total_length,
    interfaces,
    1.0,
    beam_profile,
    solve_segments,
    beam_profile_kwargs={"end_margin": end_margin},
    solve_segments_kwargs={"theta0": theta0, "end_margin": end_margin},
)




def solve_inverse_guess(d, total_length, B, interfaces, u0=None, theta0=None, end_margin=0.0, max_nfev=300):
    interfaces = [dict(interface) for interface in interfaces]
    interfaces[1]["x"] = d[0]
    interfaces[1]["y"] = d[1]

    res = solve_segments(total_length, B, interfaces, u0=u0, theta0=theta0, end_margin=end_margin, max_nfev=max_nfev)
    return res, interfaces




def make_outer_objective(total_length, B, base_interfaces, theta0=None, end_margin=0.0, max_nfev=80):
    cache = {"u0": None, "last_good_u": None}

    def objective(d):
        u0 = cache["last_good_u"] if cache["last_good_u"] is not None else cache["u0"]

        res, interfaces = solve_inverse_guess(
            d, total_length, B, base_interfaces,
            u0=u0, theta0=theta0, end_margin=end_margin, max_nfev=max_nfev,
        )

        eq_penalty = np.dot(res.fun, res.fun)

        if res.success and eq_penalty < 1e-6:
            cache["last_good_u"] = res.x.copy()

        if (not res.success) or (eq_penalty > 1e-6):
            return 1e6 + 1e3 * eq_penalty

        E = bending_energy(res.x, total_length, B, interfaces, end_margin=end_margin)
        return E + 1e4 * eq_penalty

    return objective


bounds = [(0, 2), (0, 2)]
obj = make_outer_objective(total_length, 1.0, interfaces, theta0=theta0, end_margin=end_margin, max_nfev=200)




x0 = np.array([1.575, 0.425], dtype=float)
simplex = np.array([
    x0,
    x0 + np.array([0.02, 0.00]),
    x0 + np.array([0.00, 0.02]),
])

res_outer = minimize(
    obj,
    x0=x0,
    method="Nelder-Mead",
    bounds=bounds,
    options={
        "initial_simplex": simplex,
        "xatol": 1e-5,
        "fatol": 1e-5,
        "maxfev": 200,
        "disp": True,
        "return_all": True,
    },
)

sol_outer, interfaces_outer = solve_inverse_guess(res_outer.x, total_length, 1.0, interfaces, theta0=theta0, end_margin=end_margin, max_nfev=1000)


continuity_diagnostics(sol_outer.x, total_length, 1.0, interfaces_outer, theta0, end_margin)
print("Optimal (x2, y2) =", res_outer.x)
print("inner cost =", sol_outer.cost)

plot_solution(
    sol_outer.x,
    total_length,
    interfaces_outer,
    1.0,
    beam_profile,
    solve_segments,
    beam_profile_kwargs={"end_margin": end_margin},
    solve_segments_kwargs={"theta0": theta0, "end_margin": end_margin},
)

