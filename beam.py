import numpy as  np
import matplotlib.pyplot as plt

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


def propagate_state(s_left, lengths, forces, B):
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
    
    


def solve_beam(forces, end_constraints, B, include_model=False):
    
    # Ensure spatial order so interval lengths and jump application are consistent.
    forces = sorted(forces, key=lambda force: float(force[0]) if np.ndim(force) > 0 else float(force))

    n_forces = len(forces)
    known_forces = np.zeros(n_forces, dtype=float)
    unknown_forces = np.zeros(n_forces, dtype=bool)

    def force_position(force):
        return float(force[0]) if np.ndim(force) > 0 else float(force)

    def force_value_or_none(force):
        # Convention: [x] means unknown force at x, [x, value] means known force.
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
        s_left = np.array([y_left, slope_left, 0.0, 0.0])
        f = np.asarray(u[2:2 + np.count_nonzero(unknown_forces)], dtype=float)
        current_forces = known_forces.copy()
        current_forces[unknown_forces] = f

        states = propagate_state(s_left, interval_lengths, current_forces, B)

        indexes = np.where(unknown_forces)[0]
        stable_states = [states[2 + int(i) * 2][0] for i in indexes]

        return np.array([*stable_states, states[-1][2], states[-1][3]], dtype=float)
        
    n_unknown = int(np.count_nonzero(unknown_forces))
    zero_u = np.concatenate(([0.0, 0.0], np.zeros(n_unknown)))
    r0 = residuals(zero_u)

    A = np.column_stack((
        residuals(np.concatenate(([1.0, 0.0], np.zeros(n_unknown)))) - r0,
        residuals(np.concatenate(([0.0, 1.0], np.zeros(n_unknown)))) - r0,
        *[residuals(
                np.concatenate(([0.0, 0.0], np.array([1.0 if j == i else 0.0 for j in range(n_unknown)])))
            ) - r0 for i in range(n_unknown)]
    ))
    rhs = -r0
    
    solution = np.linalg.solve(A, rhs)

    solved_forces = known_forces.copy()
    solved_forces[unknown_forces] = solution[2:2 + n_unknown]

    model = {
        "x_left": float(end_constraints[0]),
        "x_right": float(end_constraints[1]),
        "force_positions": force_positions.copy(),
        "forces": solved_forces,
        "y0": float(solution[0]),
        "m0": float(solution[1]),
        "B": float(B),
    }

    if include_model:
        return solution, A, rhs, model
    return solution, A, rhs
    

a = 3
L = 5

forces = [[-a], [2, -0.3], [a]]
end_constraints = [-L, L]

sol, A, rhs, beam_model = solve_beam(
    forces=forces,
    end_constraints=end_constraints,
    B=1.0,
    include_model=True)

print("Solution: y0 = {:.4f}, m0 = {:.4f}, unknown forces = {}".format(sol[0], sol[1], sol[2:]))
print("A matrix:\n", A)
print("RHS vector:\n", rhs)


def beam_curve(x, beam_model):
    x = np.atleast_1d(np.array(x, dtype=float))
    y = np.zeros_like(x, dtype=float)
    x_left = beam_model["x_left"]
    x_right = beam_model["x_right"]
    force_positions = np.asarray(beam_model["force_positions"], dtype=float)
    forces = np.asarray(beam_model["forces"], dtype=float)
    order = np.argsort(force_positions)
    force_positions = force_positions[order]
    forces = forces[order]
    B = float(beam_model["B"])
    s0 = np.array([beam_model["y0"], beam_model["m0"], 0.0, 0.0], dtype=float)

    for i, xi in enumerate(x):
        if xi < x_left or xi > x_right:
            y[i] = np.nan
            continue

        s = s0.copy()
        x_current = x_left

        for x_force, force in zip(force_positions, forces):
            if xi >= x_force:
                s = T_matrix(x_force - x_current) @ s
                s += force_jump(force, B)
                x_current = x_force
            else:
                break

        s = T_matrix(xi - x_current) @ s
        y[i] = s[0]

    return y[0] if len(y) == 1 else y


def beam_curve_slope(x, beam_model):
    x = np.atleast_1d(np.array(x, dtype=float))
    slope = np.zeros_like(x, dtype=float)
    x_left = beam_model["x_left"]
    x_right = beam_model["x_right"]
    force_positions = np.asarray(beam_model["force_positions"], dtype=float)
    forces = np.asarray(beam_model["forces"], dtype=float)
    order = np.argsort(force_positions)
    force_positions = force_positions[order]
    forces = forces[order]
    B = float(beam_model["B"])
    s0 = np.array([beam_model["y0"], beam_model["m0"], 0.0, 0.0], dtype=float)

    for i, xi in enumerate(x):
        if xi < x_left or xi > x_right:
            slope[i] = np.nan
            continue

        s = s0.copy()
        x_current = x_left

        for x_force, force in zip(force_positions, forces):
            if xi >= x_force:
                s = T_matrix(x_force - x_current) @ s
                s += force_jump(force, B)
                x_current = x_force
            else:
                break

        s = T_matrix(xi - x_current) @ s
        slope[i] = s[1]

    return slope[0] if len(slope) == 1 else slope



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

x_min, x_max = np.min(x_data), np.max(x_data)
y_min, y_max = np.min(y_data), np.max(y_data)

x_mid = 0.5 * (x_min + x_max)
y_mid = 0.5 * (y_min + y_max)
half_span = 0.5 * max(x_max - x_min, y_max - y_min)

ax.set_xlim(x_mid - half_span, x_mid + half_span)
ax.set_ylim(y_mid - half_span, y_mid + half_span)
ax.set_aspect("equal")
ax.grid()

ax2.grid()

plt.tight_layout()
plt.show()