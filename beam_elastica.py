import numpy as  np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares



def full_ode(s, z, B):
    # z = [x, y, theta, M, H, V] and thus function returns the derivative as a function f(z) = dz/ds. The first three components are the kinematic equations, and the last three are the equilibrium equations.
    x, y, theta, M, H, V = z
    dx = np.cos(theta)
    dy = np.sin(theta)
    dtheta = M/B
    dM = H*np.sin(theta) - V*np.cos(theta)
    dH = 0.0
    dV = 0.0
    return np.array([dx, dy, dtheta, dM, dH, dV])

def integrate_segment(z0, length, B):
    # Integrates the ODE and uses the initial conditions z0 to compute the state at the end of the segment. z(l) = z0 + integral of dz/ds from 0 to l.
    sol = solve_ivp(lambda s, z: full_ode(s, z, B), [0, length], z0, method='RK45', rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError("ODE integration failed: " + sol.message)
    return sol.y[:, -1] 

def point_force(z_l, Px, Py):
    # Computes the change in the state due to a point force applied at the end of the segment. The point force changes the internal forces H and V, but not the position or angle.
    z_new = z_l.copy()
    z_new[4] -= Px
    z_new[5] -= Py
    return z_new

def sample_segment(z0, length, B, points=200):
    s_eval = np.linspace(0.0, float(length), int(points))
    sol = solve_ivp(
        lambda s, z: full_ode(s, z, B),
        [0.0, float(length)],
        np.asarray(z0, dtype=float),
        t_eval=s_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError("ODE integration failed: " + sol.message)
    return sol.y


def build_interface_forces(segment_ends, lambda_tail, interfaces):
    lambda_tail = np.asarray(lambda_tail, dtype=float)
    lam_idx = 0
    forces = []
    for i, interface in enumerate(interfaces):
        if interface.get("type") == "con":
            theta_i = segment_ends[i][2]
            lam_i = lambda_tail[lam_idx]
            
            Px = -lam_i * np.sin(theta_i)
            Py = lam_i * np.cos(theta_i)
            
            lam_idx += 1
        
        else:
            theta_i = segment_ends[i][2]
            F = float(interface.get("force", 0.0))

            Px = -F * np.sin(theta_i)
            Py =  F * np.cos(theta_i)

        forces.append((Px, Py))
    
    return forces
        

def solved_interface_forces(solution, segment_ends, interfaces):
    n_segments = len(interfaces) + 1
    state_block = 6 * n_segments
    rho_block = n_segments - 1
    force_tail = solution[state_block + rho_block:]
    return np.array(build_interface_forces(segment_ends, force_tail, interfaces), dtype=float)

def residuals(u, total_length, B, interfaces, length_weight=0):
    n_interfaces = len(interfaces)
    n_segments = n_interfaces + 1
    state_block = 6 * n_segments
    rho_block = n_segments - 1
    lambdas_block = len([iface for iface in interfaces if iface.get("type") == "con"])

    states = [np.asarray(u[i:i+6], dtype=float) for i in range(0, 6*n_segments, 6)]
    rho = np.asarray(u[state_block: state_block + rho_block], dtype=float)
    lambdas = np.asarray(u[state_block + rho_block:], dtype=float)

    ref_lengths = reference_lengths(total_length, interfaces)
    lengths = lengths_segments(rho, total_length, ref_lengths)
    segment_ends = [integrate_segment(states[i], lengths[i], B) for i in range(n_segments)]
    forces = build_interface_forces(segment_ends, lambdas, interfaces)
    
    
    R = []
    R.extend(states[0][3:6])
    R.append(states[0][0])  # x at start should be 0

    
    for i in range(n_interfaces):
        
        x_target = float(interfaces[i].get("x", 0.0))
        R.append(segment_ends[i][0] - x_target)

        if interfaces[i].get("type") == "con":
            y_target = float(interfaces[i].get("y", 0.0))
            R.append(segment_ends[i][1] - y_target)
    
        Px, Py = forces[i]
        
        z_post = point_force(segment_ends[i], Px, Py)
        R.extend(z_post - states[i+1])
        
    R.extend(segment_ends[-1][3:6])
    
    R.extend(np.sqrt(length_weight) * (lengths - ref_lengths))
    
    return np.array(R, dtype=float)


def reference_lengths(total_length, interfaces):
    xs = [0.0] + [float(iface["x"]) for iface in interfaces] + [float(total_length)]
    xs = np.asarray(xs, dtype=float)
    if np.any(np.diff(xs) <= 0.0):
        raise ValueError("x positions must be strictly increasing inside [0, total_length].")
    return np.diff(xs)

def lengths_segments(rho, total_length, ref_lengths):
    rho = np.asarray(rho, dtype=float)
    ref_lengths = np.asarray(ref_lengths, dtype=float)

    q = np.concatenate(([1.0], np.exp(rho)))
    q = ref_lengths * q
    return float(total_length) * q / np.sum(q)

def count_unknowns(interfaces):
    return sum(1 for interface in interfaces if interface.get("type") == "con")


def beam_profile(solution, total_length, interfaces, B, points_per_segment=200):
    n_interfaces = len(interfaces)
    n_segments = n_interfaces + 1
    state_block = 6 * n_segments
    rho = np.asarray(solution[state_block: state_block + n_segments - 1], dtype=float)
    lengths = lengths_segments(rho, total_length, reference_lengths(total_length, interfaces))
    
    
    states = [np.asarray(solution[i:i+6], dtype=float) for i in range(0, 6*n_segments, 6)]

    sampled_segments = [
        sample_segment(states[i], lengths[i], B, points=points_per_segment)
        for i in range(n_segments)
    ]

    segment_ends = [seg[:, -1] for seg in sampled_segments]

    x = np.concatenate([seg[0] for seg in sampled_segments])
    y = np.concatenate([seg[1] for seg in sampled_segments])
    theta = np.concatenate([seg[2] for seg in sampled_segments])
    
    forces = solved_interface_forces(solution, segment_ends, interfaces)

    joints = np.array(
        [[seg[0][-1], seg[1][-1]] for seg in sampled_segments[:-1]],
        dtype=float,
    )

    return x, y, theta, joints, forces


class DraggablePoints:
    def __init__(self, fig, ax, scatter, interfaces, total_length, B, replot_callback=None):
        self.fig = fig
        self.ax = ax
        self.scatter = scatter
        self.points = np.asarray(scatter.get_offsets(), dtype=float).copy()
        self.active_index = None
        self.drag_start = None
        
        self.interfaces = interfaces
        self.total_length = total_length
        self.B = B
        self.replot_callback = replot_callback

        self._cid_press = fig.canvas.mpl_connect('button_press_event', self.on_press)
        self._cid_move = fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self._cid_release = fig.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Find closest point
        distances = np.hypot(self.points[:, 0] - event.xdata, self.points[:, 1] - event.ydata)
        self.active_index = int(np.argmin(distances))
        dist = distances[self.active_index]
        
        if dist > 0.2:  # Tolerance for clicking near a point
            self.active_index = None
            return
        
        self.drag_start = np.array([event.xdata, event.ydata], dtype=float)

    def on_motion(self, event):
        if self.active_index is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        current_pos = np.array([event.xdata, event.ydata], dtype=float)
        delta = current_pos - self.drag_start
        self.points[self.active_index] += delta
        self.drag_start = current_pos
        
        self.scatter.set_offsets(self.points)
        self.fig.canvas.draw()

    def on_release(self, event):
        if self.active_index is None:
            return

        idx = self.active_index
        new_x = self.points[idx, 0]
        new_y = self.points[idx, 1]
        
        # Update interface position
        self.interfaces[idx]["x"] = float(new_x)
        if self.interfaces[idx].get("type") == "con":
            self.interfaces[idx]["y"] = float(new_y)
        
        # Re-solve if callback is provided
        if self.replot_callback is not None:
            try:
                self.replot_callback(self.interfaces)
            except Exception as e:
                print(f"Error re-solving: {e}")
                import traceback
                traceback.print_exc()
        
        self.active_index = None
        self.drag_start = None


def enable_interactive_points(fig, ax, scatter, interfaces, total_length, B, replot_callback=None):
    return DraggablePoints(fig, ax, scatter, interfaces, total_length, B, replot_callback=replot_callback)

def plot_solution(solution, total_length, interfaces, B):
    x, y, theta, joints, forces = beam_profile(
        solution,
        total_length,
        interfaces,
        B,
        points_per_segment=400,
    )

    print("All joint forces (Px, Py):")
    for i, (px, py) in enumerate(forces, start=1):
        print(f"  joint {i}: Px={px:.6g}, Py={py:.6g}")
    

    fig, ax = plt.subplots(figsize=(12, 10))

    line, = ax.plot(x, y, color='blue', label='Beam deflection')

    point_colors = ['red' if iface.get("type") == "con" else 'green' for iface in interfaces]
    point_sizes = [60 if iface.get("type") == "con" else 40 for iface in interfaces]
    joint_scatter = ax.scatter(
        joints[:, 0],
        joints[:, 1],
        c=point_colors,
        s=point_sizes,
        zorder=3,
        marker='o',
    )
    
    # Initial arrow collection
    arrows = []
    max_force = np.max(np.linalg.norm(joints - forces, axis=1))
    if max_force <= 0.0:
        max_force = 1.0
    for i in range(len(joints)):
        arrow = ax.arrow(joints[i, 0], joints[i, 1], forces[i, 0]/max_force, forces[i, 1]/max_force, 
                         color='orange', width=0.02, head_width=0.1)
        arrows.append(arrow)
    
    ax.set_ylabel('Deflection (y)', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    
    x_data = np.concatenate([x, joints[:, 0]])
    y_data = np.concatenate([y, joints[:, 1]])

    finite_xy = np.isfinite(x_data) & np.isfinite(y_data)
    if np.any(finite_xy):
        x_plot = x_data[finite_xy]
        y_plot = y_data[finite_xy]
        x_min, x_max = np.min(x_plot), np.max(x_plot)
        y_min, y_max = np.min(y_plot), np.max(y_plot)
    else:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = -1.0, 1.0

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    half_span = 0.5 * max(x_max - x_min, y_max - y_min)
    if not np.isfinite(half_span) or half_span <= 0.0:
        half_span = 1.0

    ax.set_xlim(x_mid - half_span, x_mid + half_span)
    ax.set_ylim(y_mid - half_span, y_mid + half_span)
    ax.set_aspect('equal')
    ax.grid()
    ax.legend()
    
    def replot_callback(updated_interfaces):
        """Re-solve and update the plot when points are dragged."""
        nonlocal arrows
        
        # Re-solve with updated interfaces
        sol_new = solve_segments(total_length, B, updated_interfaces)
        x_new, y_new, theta_new, joints_new, forces_new = beam_profile(
            sol_new.x,
            total_length,
            updated_interfaces,
            B,
            points_per_segment=400,
        )
        
        # Update line
        line.set_data(x_new, y_new)
        
        # Update scatter points
        joint_scatter.set_offsets(joints_new)
        
        # Remove old arrows
        for arrow in arrows:
            arrow.remove()
        arrows.clear()
        
        # Add new arrows
        max_force_new = np.max(np.linalg.norm(joints_new - forces_new, axis=1))
        if max_force_new <= 0.0:
            max_force_new = 1.0
        for i in range(len(joints_new)):
            arrow = ax.arrow(joints_new[i, 0], joints_new[i, 1], 
                            forces_new[i, 0]/max_force_new, forces_new[i, 1]/max_force_new, 
                            color='orange', width=0.02, head_width=0.1)
            arrows.append(arrow)
        
        # Update axis limits
        x_data_new = np.concatenate([x_new, joints_new[:, 0]])
        y_data_new = np.concatenate([y_new, joints_new[:, 1]])
        finite_xy_new = np.isfinite(x_data_new) & np.isfinite(y_data_new)
        if np.any(finite_xy_new):
            x_plot_new = x_data_new[finite_xy_new]
            y_plot_new = y_data_new[finite_xy_new]
            x_min_new, x_max_new = np.min(x_plot_new), np.max(x_plot_new)
            y_min_new, y_max_new = np.min(y_plot_new), np.max(y_plot_new)
        else:
            x_min_new, x_max_new = float(np.min(x_new)), float(np.max(x_new))
            y_min_new, y_max_new = -1.0, 1.0
        
        x_mid_new = 0.5 * (x_min_new + x_max_new)
        y_mid_new = 0.5 * (y_min_new + y_max_new)
        half_span_new = 0.5 * max(x_max_new - x_min_new, y_max_new - y_min_new)
        if not np.isfinite(half_span_new) or half_span_new <= 0.0:
            half_span_new = 1.0
        
        ax.set_xlim(x_mid_new - half_span_new, x_mid_new + half_span_new)
        ax.set_ylim(y_mid_new - half_span_new, y_mid_new + half_span_new)
        
        print("Updated interface forces:")
        for i, (px, py) in enumerate(forces_new, start=1):
            print(f"  joint {i}: Px={px:.6g}, Py={py:.6g}")
        
        fig.canvas.draw()
    
    draggable = enable_interactive_points(fig, ax, joint_scatter, interfaces, total_length, B, replot_callback=replot_callback)
    fig.draggable = draggable  # Keep a reference so it doesn't get garbage collected

    plt.show()

def initial_guess(total_length, interfaces):
    ref_lengths = reference_lengths(total_length, interfaces)
    n_segments = len(ref_lengths)
    
    xs = [0.0]
    for L in ref_lengths[:-1]:
        xs.append(xs[-1] + L)
    xs = np.asarray(xs, dtype=float)
    
    states = []
    for i in range(n_segments):
        states.extend([xs[i], 0.0, 0.0, 0.0, 0.0, 0.0])
    
    eta = np.zeros(n_segments, dtype=float)
    n_unknowns = count_unknowns(interfaces)
    lambdas = np.zeros(n_unknowns, dtype=float)

    return np.concatenate([np.array(states, dtype=float), eta, lambdas])

def solve_segments(total_length, B, interfaces):
    u0 = initial_guess(total_length, interfaces)
    return least_squares(residuals, u0, args=(total_length, B, interfaces), method='trf', xtol=1e-10, ftol=1e-10, gtol=1e-10)

def continuity_diagnostics(solution, total_length, B, interfaces):
    n_interfaces = len(interfaces)
    n_segments = n_interfaces + 1
    state_block = 6 * n_segments
    rho_block = n_segments - 1

    states = [np.asarray(solution[i:i+6], dtype=float) for i in range(0, 6*n_segments, 6)]
    rho = np.asarray(solution[state_block: state_block + rho_block], dtype=float)
    lambdas = np.asarray(solution[state_block + rho_block:], dtype=float)

    ref_lengths = reference_lengths(total_length, interfaces)
    lengths = lengths_segments(rho, total_length, ref_lengths)
    segment_ends = [integrate_segment(states[i], lengths[i], B) for i in range(n_segments)]
    forces = build_interface_forces(segment_ends, lambdas, interfaces)

    for i in range(n_interfaces):
        z_post = point_force(segment_ends[i], *forces[i])
        mismatch = z_post - states[i+1]
        print(f"joint {i+1}: norm = {np.linalg.norm(mismatch):.6e}, mismatch = {mismatch}")

    r = residuals(solution, total_length, B, interfaces)
    print("max abs residual =", np.max(np.abs(r)))
    print("residual norm    =", np.linalg.norm(r))

total_length = 6.0

interfaces = [
    {"x": 0.5,  "type": "con",  "y": 0.0},
    {"x": 1.0,  "type": "load", "force": -2},
    {"x": 3.5,  "type": "load", "force": 2},
    {"x": 4.0,  "type": "con",  "y": 1.0},
]

sol = solve_segments(total_length, 1.0, interfaces=interfaces)

continuity_diagnostics(sol.x, total_length, 1.0, interfaces)

plot_solution(sol.x, total_length, interfaces, 1.0)


