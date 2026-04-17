import numpy as np
import matplotlib.pyplot as plt

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

    def sync_points_from_scatter(self):
        self.points = np.asarray(self.scatter.get_offsets(), dtype=float).copy()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Keep drag hit-testing in sync with the latest plotted points.
        self.sync_points_from_scatter()

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
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.active_index is None:
            return

        idx = self.active_index
        new_x = self.points[idx, 0]
        new_y = self.points[idx, 1]
        interface = self.interfaces[idx]
        
        # Update interface position
        if "axis" in interface:
            if interface["axis"] == "x":
                interface["value"] = float(new_x)
            elif interface["axis"] == "y":
                interface["value"] = float(new_y)
            else:
                raise ValueError("Interface axis must be 'x' or 'y'.")
        else:
            interface["x"] = float(new_x)
            if interface.get("type") == "con":
                interface["y"] = float(new_y)
        
        # Re-solve if callback is provided
        if self.replot_callback is not None:
            try:
                self.replot_callback(self.interfaces)
                self.sync_points_from_scatter()
            except Exception as e:
                print(f"Error re-solving: {e}")
                import traceback
                traceback.print_exc()
        
        self.active_index = None
        self.drag_start = None


def enable_interactive_points(fig, ax, scatter, interfaces, total_length, B, replot_callback=None):
    return DraggablePoints(fig, ax, scatter, interfaces, total_length, B, replot_callback=replot_callback)

def plot_solution(
    solution,
    total_length,
    interfaces,
    B,
    beam_profile,
    solve_segments,
    beam_profile_kwargs=None,
    solve_segments_kwargs=None,
):
    if beam_profile_kwargs is None:
        beam_profile_kwargs = {}
    if solve_segments_kwargs is None:
        solve_segments_kwargs = {}

    x, y, theta, joints, forces = beam_profile(
        solution,
        total_length,
        interfaces,
        B,
        points_per_segment=400,
        **beam_profile_kwargs,
    )

    print("All joint forces (Px, Py):")
    for i, (px, py) in enumerate(forces, start=1):
        if interfaces[i - 1].get("type") == "load":
            mode = "Py" if "Py" in interfaces[i - 1] else "force"
            print(f"  joint {i} [{mode}]: Px={px:.6g}, Py={py:.6g},     locations: x={joints[i - 1, 0]:.6g}, y={joints[i - 1, 1]:.6g}")
        else:
            print(f"  joint {i} [con]: Px={px:.6g}, Py={py:.6g},     locations: x={joints[i - 1, 0]:.6g}, y={joints[i - 1, 1]:.6g}")

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
        sol_new = solve_segments(total_length, B, updated_interfaces, **solve_segments_kwargs)
        x_new, y_new, theta_new, joints_new, forces_new = beam_profile(
            sol_new.x,
            total_length,
            updated_interfaces,
            B,
            points_per_segment=400,
            **beam_profile_kwargs,
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
        
        fig.canvas.draw()
    
    draggable = enable_interactive_points(fig, ax, joint_scatter, interfaces, total_length, B, replot_callback=replot_callback)
    fig.draggable = draggable  # Keep a reference so it doesn't get garbage collected

    plt.show()
