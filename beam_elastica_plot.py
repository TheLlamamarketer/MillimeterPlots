import numpy as np
import matplotlib.pyplot as plt

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

    print("Interface forces and positions:")
    print(f"theta[0] = {theta[0]*180/np.pi:.4g}°, theta[-1] = {theta[-1]*180/np.pi:.4g}°")
    for i, (px, py) in enumerate(forces, start=1):
        mode = "con" if interfaces[i - 1].get("type") == "con" else "load"
        print(f"  joint {i} [{mode}]: Px={px:6.4g}, Py={py:6.4g}, x={joints[i - 1, 0]:10.4g}, y={joints[i - 1, 1]:10.4g}")

    fig, ax = plt.subplots(figsize=(12, 10))

    line, = ax.plot(x, y, color='blue')

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
    half_span += 0.2  # Ensure at least a small visual margin around the data.

    ax.set_xlim(x_mid - half_span, x_mid + half_span)
    ax.set_ylim(y_mid - half_span, y_mid + half_span)
    ax.set_aspect('equal')
    ax.grid()
    #ax.legend()
    
    plt.show()
