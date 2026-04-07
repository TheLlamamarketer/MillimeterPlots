import numpy as  np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, root
from numpy.polynomial.legendre import leggauss

def curve_init(x):
    return np.zeros_like(x)

def T_matrix(l):
    return np.array([
        [1, l, l**2/2, l**3/6],
        [0, 1, l, l**2/2],
        [0, 0, 1, l],
        [0, 0, 0, 1]
    ])

def force_jump(force, B):
    return np.array([0, 0, 0, force/B])


def propagate_state_linear(s_left, lengths, forces, B):
    s = s_left.copy()
    states = [s_left.copy()]

    n = len(forces)
    for i in range(n):
        s = T_matrix(lengths[i]) @ s
        states.append(s.copy())

        s += force_jump(forces[i], B)
        states.append(s.copy())

    states.append(T_matrix(lengths[-1]) @ s)
    return states

def segment_length(s_left, length, quad_order=16):
    y0, g0, k0, q0 = s_left
    xi, wi = leggauss(quad_order)
    xi = 0.5 * (xi + 1) * length
    
    g_xi = g0 + k0 * xi + 0.5 * q0 * xi**2

    if np.any(np.abs(g_xi) >= 1.0):
        raise ValueError("Nonphysical state reached: |g| >= 1")

    integrand = 1.0 / np.sqrt(1.0 - g_xi**2)
    return np.sum(wi * integrand) * 0.5 * length

def total_length(s_left, lengths, forces, B, quad_order=16):
    s = s_left.copy()
    total_len = 0.0
    
    n = len(forces)
    for i in range(n):
        seg_len = segment_length(s, lengths[i], quad_order=quad_order)
        total_len += seg_len
        s = propagate_segment(s, lengths[i], quad_order=quad_order)
        s+= force_jump(forces[i], B)

    total_len += segment_length(s, lengths[-1], quad_order=quad_order)
    return total_len

def solve_beam_linearized(forces, end_constraints, B):
    forces = sorted(forces, key=lambda f: float(f[0]) if np.ndim(f) > 0 else float(f))

    n_forces = len(forces)
    known_forces = np.zeros(n_forces, dtype=float)
    unknown_forces = np.zeros(n_forces, dtype=bool)

    def force_position(force):
        return float(force[0]) if np.ndim(force) > 0 else float(force)

    def force_value_or_none(force):
        if np.ndim(force) == 0:
            return None
        return float(force[1]) if len(force) >= 2 else None

    force_positions = np.array([force_position(force) for force in forces], dtype=float)
    interval_lengths = [force_positions[0] - end_constraints[0]]
    for i in range(1, n_forces):
        interval_lengths.append(force_positions[i] - force_positions[i - 1])
    interval_lengths.append(end_constraints[1] - force_positions[-1])

    for i, force in enumerate(forces):
        value = force_value_or_none(force)
        if value is None:
            unknown_forces[i] = True
        else:
            known_forces[i] = value

    def residuals(u):
        y_left, slope_left = u[0], u[1]
        s_left = np.array([y_left, slope_left, 0.0, 0.0], dtype=float)
        f_unknown = np.asarray(u[2:2 + np.count_nonzero(unknown_forces)], dtype=float)
        current_forces = known_forces.copy()
        current_forces[unknown_forces] = f_unknown

        states = propagate_state_linear(s_left, interval_lengths, current_forces, B)
        idx_unknown = np.where(unknown_forces)[0]
        contact_y = [states[2 + int(i) * 2][0] for i in idx_unknown]
        return np.array([*contact_y, states[-1][2], states[-1][3]], dtype=float)

    n_unknown = int(np.count_nonzero(unknown_forces))
    u_zero = np.concatenate(([0.0, 0.0], np.zeros(n_unknown)))
    r0 = residuals(u_zero)

    cols = [
        residuals(np.concatenate(([1.0, 0.0], np.zeros(n_unknown)))) - r0,
        residuals(np.concatenate(([0.0, 1.0], np.zeros(n_unknown)))) - r0,
    ]
    for i in range(n_unknown):
        unit_f = np.array([1.0 if j == i else 0.0 for j in range(n_unknown)], dtype=float)
        cols.append(residuals(np.concatenate(([0.0, 0.0], unit_f))) - r0)

    A = np.column_stack(cols)
    rhs = -r0
    solution = np.linalg.solve(A, rhs)
    return solution


def propagate_segment(s_left, length, quad_order=16):
    y0, g0, k0, q0 = s_left
    l = float(length)

    # right-end kappa and g are analytic
    k1 = k0 + q0 * l
    g1 = g0 + k0 * l + 0.5 * q0 * l**2
    
    xi, wi = leggauss(quad_order)
    xi = 0.5 * (xi + 1) * l  # Map from [-1, 1] to [0, l]
    
    g_xi = g0 + k0 * xi + 0.5 * q0 * xi**2
    
    if abs(g1) >= 1.0 or np.any(np.abs(g_xi) >= 1.0):
        raise ValueError("Nonphysical state reached: |g| >= 1")
    
    integrand = g_xi / np.sqrt(1.0 - g_xi**2)
    y1 = y0 + 0.5 * l * np.sum(wi * integrand)
    
    
    return np.array([y1, g1, k1, q0], dtype=float)
    
def propagate_state(s_left, lengths, forces, B, quad_order=16):
    s = s_left.copy()
    states = [s_left.copy()]
    
    n = len(forces)
    for i in range(n):
        s = propagate_segment(s, lengths[i], quad_order=quad_order)
        states.append(s.copy())
        
        s += force_jump(forces[i], B)
        states.append(s.copy())
    
    states.append(propagate_segment(s, lengths[-1], quad_order=quad_order))
    return states

def solve_beam(forces, end_constraints, B, L0=None, include_model=False, quad_order=16):
    """
    forces:
        [x]         -> unknown force at x
        [x, value]  -> known force at x
    end_constraints = [x_left, x_right]

    Left boundary assumption:
        kappa = 0, q = 0
    Unknowns solved for:
        y_left, g_left, unknown contact forces

    Residuals:
        y(contact_i) = 0 for unknown-force contacts
        kappa(right end) = 0
        q(right end) = 0
    """
    forces = sorted(forces, key=lambda f: float(f[0]) if np.ndim(f) > 0 else float(f))

    n_forces = len(forces)
    known_forces = np.zeros(n_forces, dtype=float)
    unknown_forces = np.zeros(n_forces, dtype=bool)

    def force_position(force):
        return float(force[0]) if np.ndim(force) > 0 else float(force)

    def force_value_or_none(force):
        if np.ndim(force) == 0:
            return None
        return float(force[1]) if len(force) >= 2 else None

    force_positions = np.array([force_position(f) for f in forces], dtype=float)
    x_mid_target = 0.5 * float(end_constraints[0] + end_constraints[1])
    if L0 is None:
        L0 = float(end_constraints[1] - end_constraints[0])

    for i, force in enumerate(forces):
        value = force_value_or_none(force)
        if value is None:
            unknown_forces[i] = True
        else:
            known_forces[i] = value

    def residuals(u):
        y_left, g_left, x_left, x_right = u[0], u[1], u[2], u[3]
        f_unknown = np.asarray(u[4:4 + np.count_nonzero(unknown_forces)], dtype=float)

        if x_left >= force_positions[0] or x_right <= force_positions[-1] or x_right <= x_left:
            return np.full(4 + np.count_nonzero(unknown_forces), 1e3)

        interval_lengths = [force_positions[0] - x_left]
        for i in range(1, n_forces):
            interval_lengths.append(force_positions[i] - force_positions[i - 1])
        interval_lengths.append(x_right - force_positions[-1])

        current_forces = known_forces.copy()
        current_forces[unknown_forces] = f_unknown

        # left free tail: kappa = 0, q = 0
        s_left = np.array([y_left, g_left, 0.0, 0.0], dtype=float)

        try:
            states = propagate_state(
                s_left, interval_lengths, current_forces, B, quad_order=quad_order
            )
            L = total_length(s_left, interval_lengths, current_forces, B, quad_order=quad_order)
        except ValueError:
            # penalize nonphysical regions where |g| >= 1
            return np.full(4 + np.count_nonzero(unknown_forces), 1e3)

        len_res = L - L0
        mid_res = 0.5 * (x_left + x_right) - x_mid_target

        # enforce active unknown contacts to lie on y=0
        idx_unknown = np.where(unknown_forces)[0]
        contact_y = [states[2 + int(i) * 2][0] for i in idx_unknown]

        # right free tail: kappa = 0, q = 0
        end_kappa = states[-1][2]
        end_q = states[-1][3]

        return np.array([*contact_y, end_kappa, end_q, len_res, mid_res], dtype=float)

    n_unknown = int(np.count_nonzero(unknown_forces))

    # Warm-start with a linearized beam solution; this is usually close enough
    # to keep the nonlinear solve in the physical branch.
    u0 = np.concatenate(([0.0, 0.0, end_constraints[0], end_constraints[1]], np.zeros(n_unknown)))
    lin_solution = None
    try:
        lin_solution = solve_beam_linearized(forces, end_constraints, B)
        y0_lin = float(lin_solution[0])
        slope_lin = float(lin_solution[1])
        # g = dy/ds = slope / sqrt(1 + slope^2), constrained to (-1, 1)
        g0_lin = slope_lin / np.sqrt(1.0 + slope_lin**2)
        g0_lin = float(np.clip(g0_lin, -0.95, 0.95))
        f_lin = np.asarray(lin_solution[2:2 + n_unknown], dtype=float)
        u0 = np.concatenate(([y0_lin, g0_lin, end_constraints[0], end_constraints[1]], f_lin))
    except Exception:
        # Keep zero initial guess if linear warm-start is unavailable.
        pass

    sol = root(residuals, u0, method="hybr")
    if sol.success:
        solution = sol.x
    else:
        # Fallback to a bounded least-squares solve, then refine with hybr.
        lower = np.full(4 + n_unknown, -np.inf, dtype=float)
        upper = np.full(4 + n_unknown, np.inf, dtype=float)
        lower[1] = -0.98
        upper[1] = 0.98
        upper[2] = force_positions[0] - 1e-9
        lower[3] = force_positions[-1] + 1e-9

        lsq = least_squares(
            residuals,
            u0,
            bounds=(lower, upper),
            method="trf",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=20000,
        )
        if not lsq.success:
            raise RuntimeError(
                f"Nonlinear solve failed: {sol.message}; least_squares fallback also failed: {lsq.message}"
            )

        sol_refine = root(residuals, lsq.x, method="hybr")
        if sol_refine.success:
            solution = sol_refine.x
        else:
            # Accept bounded nonlinear LSQ solution only if residual is small.
            lsq_res_norm = np.linalg.norm(lsq.fun, ord=2)
            if lsq_res_norm > 1e-7:
                raise RuntimeError(
                    f"Nonlinear solve failed: {sol.message}; nonlinear fallback residual too large: {lsq_res_norm:.3e}"
                )
            solution = lsq.x

    solved_forces = known_forces.copy()
    solved_forces[unknown_forces] = solution[4:4 + n_unknown]

    model = {
        "x_left": float(solution[2]),
        "x_right": float(solution[3]),
        "force_positions": force_positions.copy(),
        "forces": solved_forces,
        "y_left": float(solution[0]),
        "g_left": float(solution[1]),
        "B": float(B),
        "quad_order": int(quad_order),
    }

    if include_model:
        return solution, model
    return solution

def beam_curve(x, beam_model):
    x = np.atleast_1d(np.array(x, dtype=float))
    y = np.zeros_like(x, dtype=float)

    x_left = beam_model["x_left"]
    x_right = beam_model["x_right"]
    force_positions = np.asarray(beam_model["force_positions"], dtype=float)
    forces = np.asarray(beam_model["forces"], dtype=float)
    B = float(beam_model["B"])
    quad_order = int(beam_model["quad_order"])

    s0 = np.array([beam_model["y_left"], beam_model["g_left"], 0.0, 0.0], dtype=float)

    for i, xi in enumerate(x):
        if xi < x_left or xi > x_right:
            y[i] = np.nan
            continue

        s = s0.copy()
        x_current = x_left

        for x_force, force in zip(force_positions, forces):
            if xi >= x_force:
                s = propagate_segment(s, x_force - x_current, quad_order=quad_order)
                s += force_jump(force, B)
                x_current = x_force
            else:
                break

        s = propagate_segment(s, xi - x_current, quad_order=quad_order)
        y[i] = s[0]

    return y[0] if len(y) == 1 else y


def beam_curve_slope(x, beam_model):
    x = np.atleast_1d(np.array(x, dtype=float))
    slope = np.zeros_like(x, dtype=float)

    x_left = beam_model["x_left"]
    x_right = beam_model["x_right"]
    force_positions = np.asarray(beam_model["force_positions"], dtype=float)
    forces = np.asarray(beam_model["forces"], dtype=float)
    B = float(beam_model["B"])
    quad_order = int(beam_model["quad_order"])

    s0 = np.array([beam_model["y_left"], beam_model["g_left"], 0.0, 0.0], dtype=float)

    for i, xi in enumerate(x):
        if xi < x_left or xi > x_right:
            slope[i] = np.nan
            continue

        s = s0.copy()
        x_current = x_left

        for x_force, force in zip(force_positions, forces):
            if xi >= x_force:
                s = propagate_segment(s, x_force - x_current, quad_order=quad_order)
                s += force_jump(force, B)
                x_current = x_force
            else:
                break

        s = propagate_segment(s, xi - x_current, quad_order=quad_order)

        g = s[1]
        slope[i] = g / np.sqrt(1.0 - g**2)

    return slope[0] if len(slope) == 1 else slope

a = 4
L = 5

forces = [[-a, 1],[-2, -0.9], [0, 1], [2, -0.9], [a, 1]]
end_constraints = [-L, L]

sol, beam_model = solve_beam(
    forces=forces,
    end_constraints=end_constraints,
    B=1.0, L0=10, include_model=True, quad_order=32)

print("Solution: y_left = {:.4f}, g_left = {:.4f}, x_left = {:.4f}, x_right = {:.4f}, unknown forces = {}".format(sol[0], sol[1], sol[2], sol[3], sol[4:]))


constraints = []
for i, force in enumerate(forces):
    x_pos = force[0]
    y_pos = beam_curve(x_pos, beam_model)
    direction = -1 if len(force) == 1 else -np.sign(float(force[1]))
    constraints.append(np.array([x_pos, y_pos, direction]))

x = np.linspace(-L, L, 1000)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left subplot: Deflection
ax.plot(x, curve_init(x), linestyle='--', label='Initial curve')
ax.plot(x, beam_curve(x, beam_model), color='blue', label='Beam deflection')
ax.set_ylabel('Deflection (y)', color='blue')
ax.tick_params(axis='y', labelcolor='blue')

# Right subplot: Slope
ax2.plot(x, beam_curve_slope(x, beam_model), color='red', linestyle='--', label='Beam slope')
ax2.set_ylabel('Slope (dy/dx)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

def triangle_vertices(x0, y0, direction, width=0.25, height=0.12):
    return np.array([
        [x0, y0],
        [x0 - width / 2, y0 + direction * 2*height],
        [x0 + width / 2, y0 + direction * 2*height],
    ])

for p in constraints:
    x0, y0, direction = p
    verts = triangle_vertices(x0, y0, int(direction))
    triangle = plt.Polygon(verts, closed=True, facecolor='green', edgecolor='black', alpha=0.6)
    ax.add_patch(triangle)

ax.set_xlabel('x')
ax.set_title('Beam Deflection')

ax2.set_xlabel('x')
ax2.set_title('Beam Slope')

# Calculate limits based on deflection data only
x_data = np.concatenate([x, np.array([p[0] for p in constraints])])
y_data = np.concatenate([beam_curve(x, beam_model), curve_init(x), np.array([p[1] for p in constraints])])

# Match x/y samples and ignore non-finite values from out-of-domain points.
y_curve = beam_curve(x, beam_model)
y_zero = curve_init(x)
x_curve = np.concatenate([x, x])
y_curve_all = np.concatenate([y_curve, y_zero])

x_data = np.concatenate([x_curve, np.array([p[0] for p in constraints])])
y_data = np.concatenate([y_curve_all, np.array([p[1] for p in constraints])])

finite_xy = np.isfinite(x_data) & np.isfinite(y_data)
if np.any(finite_xy):
    x_plot = x_data[finite_xy]
    y_plot = y_data[finite_xy]
    x_min, x_max = np.min(x_plot), np.max(x_plot)
    y_min, y_max = np.min(y_plot), np.max(y_plot)
else:
    x_min, x_max = float(end_constraints[0]), float(end_constraints[1])
    y_min, y_max = -1.0, 1.0

x_mid = 0.5 * (x_min + x_max)
y_mid = 0.5 * (y_min + y_max)
half_span = 0.5 * max(x_max - x_min, y_max - y_min)
if not np.isfinite(half_span) or half_span <= 0.0:
    half_span = 1.0

ax.set_xlim(x_mid - half_span, x_mid + half_span)
ax.set_ylim(y_mid - half_span, y_mid + half_span)
ax.set_aspect("equal")
ax.grid()

ax2.grid()

plt.tight_layout()
plt.show()