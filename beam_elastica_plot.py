import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def fmt_float(value, width=11, fixed=6, sci=3, small=1e-4, large=1e5):
    value = float(value)
    if not np.isfinite(value):
        return f"{value:>{width}}"
    if value != 0.0 and (abs(value) < small or abs(value) >= large):
        return f"{value:{width}.{sci}e}"
    return f"{value:{width}.{fixed}f}"

def force_directions(forces, interfaces=None, mirror_y=False, tol=1e-12):
    directions = np.zeros_like(forces, dtype=float)
    for i, force in enumerate(forces):
        vec = np.array(force, dtype=float)
        if mirror_y:
            vec[1] *= -1.0

        norm = float(np.hypot(vec[0], vec[1]))
        if norm <= tol and interfaces is not None and "normal" in interfaces[i]:
            vec = np.asarray(interfaces[i]["normal"], dtype=float)
            if mirror_y:
                vec[1] *= -1.0
            norm = float(np.hypot(vec[0], vec[1]))

        if norm > tol:
            directions[i] = vec / norm

    return directions

def save_mirrored_pdf(
    solution,
    total_length,
    interfaces,
    B,
    beam_profile,
    path="beam_mirrored.pdf",
    beam_profile_kwargs=None,
    unit_mm=1.0,
    margin=1.0,
    line_width=0.2,
    beam_radius=None,
    point_size=1.2,
    contact_radius=None,
    points_per_segment=2000,
    beam_color="blue",
    contact_color="red",
    mirror_beam_color=None,
    mirror_contact_color=None,
    beam_center_offset=0.0,
    contact_force_offset=0.0,
    mirror_contact_tol=1e-9,
    skip_unoffset_axis_duplicates=True,
):
    if beam_profile_kwargs is None:
        beam_profile_kwargs = {}

    x, y, theta, joints, forces = beam_profile(
        solution,
        total_length,
        interfaces,
        B,
        points_per_segment=points_per_segment,
        **beam_profile_kwargs,
    )

    unit_mm = float(unit_mm)
    beam_radius = None if beam_radius is None else float(beam_radius)
    contact_radius = None if contact_radius is None else float(contact_radius)
    beam_center_offset = float(beam_center_offset)
    contact_force_offset = float(contact_force_offset)
    if mirror_beam_color is None:
        mirror_beam_color = beam_color
    if mirror_contact_color is None:
        mirror_contact_color = contact_color

    upper_y = y + beam_center_offset
    mirrored_y = -y - beam_center_offset

    upper_joints = joints.copy()
    upper_joints[:, 1] += beam_center_offset
    mirrored_joints = joints.copy()
    mirrored_joints[:, 1] = -mirrored_joints[:, 1] - beam_center_offset

    if contact_force_offset != 0.0:
        upper_joints += contact_force_offset * force_directions(forces, interfaces=interfaces, mirror_y=False)
        mirrored_joints += contact_force_offset * force_directions(forces, interfaces=interfaces, mirror_y=True)
    contact_points = upper_joints.copy()
    if contact_radius is not None and contact_force_offset != 0.0:
        contact_points -= np.sign(contact_force_offset) * contact_radius * force_directions(forces, interfaces=interfaces)
    print("Contact surface positions:")
    for i, (cx, cy) in enumerate(contact_points, start=1):
        print(f"  contact {i:>2}: x={fmt_float(cx)}, y={fmt_float(cy)}")

    # Avoid drawing duplicate mirrored copies of contacts that lie on the mirror axis.
    if skip_unoffset_axis_duplicates and beam_center_offset == 0.0 and contact_force_offset == 0.0:
        mirror_mask = np.abs(joints[:, 1]) > mirror_contact_tol
    else:
        mirror_mask = np.ones(len(joints), dtype=bool)

    x_data = np.concatenate([x, x, upper_joints[:, 0], mirrored_joints[mirror_mask, 0]])
    y_data = np.concatenate([upper_y, mirrored_y, upper_joints[:, 1], mirrored_joints[mirror_mask, 1]])
    finite = np.isfinite(x_data) & np.isfinite(y_data)
    if not np.any(finite):
        raise ValueError("Cannot export mirrored PDF because no finite geometry was produced.")

    physical_radius = max(
        0.0 if beam_radius is None else beam_radius,
        0.0 if contact_radius is None else contact_radius,
    )
    drawing_margin = float(margin) + physical_radius

    x_min = float(np.min(x_data[finite])) - drawing_margin
    x_max = float(np.max(x_data[finite])) + drawing_margin
    y_min = float(np.min(y_data[finite])) - drawing_margin
    y_max = float(np.max(y_data[finite])) + drawing_margin

    width_units = x_max - x_min
    height_units = y_max - y_min
    if width_units <= 0.0 or height_units <= 0.0:
        raise ValueError("Cannot export mirrored PDF with non-positive plotted size.")

    mm_per_inch = 25.4
    if beam_radius is None:
        beam_line_width = line_width
    else:
        beam_line_width = 2.0 * beam_radius * unit_mm * 72.0 / mm_per_inch

    old_path_simplify = plt.rcParams["path.simplify"]
    plt.rcParams["path.simplify"] = False
    fig = plt.figure(
        figsize=(width_units * unit_mm / mm_per_inch, height_units * unit_mm / mm_per_inch),
        frameon=False,
    )
    try:
        fig.patch.set_alpha(0.0)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.patch.set_alpha(0.0)

        ax.plot(
            x,
            upper_y,
            color=beam_color,
            linewidth=beam_line_width,
            solid_capstyle="butt",
            solid_joinstyle="round",
        )
        ax.plot(
            x,
            mirrored_y,
            color=mirror_beam_color,
            linewidth=beam_line_width,
            solid_capstyle="butt",
            solid_joinstyle="round",
        )
        if contact_radius is None:
            ax.scatter(upper_joints[:, 0], upper_joints[:, 1], color=contact_color, s=point_size, zorder=3)
            ax.scatter(
                mirrored_joints[mirror_mask, 0],
                mirrored_joints[mirror_mask, 1],
                color=mirror_contact_color,
                s=point_size,
                zorder=3,
            )
        else:
            for cx, cy in upper_joints:
                ax.add_patch(Circle((cx, cy), contact_radius, facecolor=contact_color, edgecolor="none", zorder=3))
            for cx, cy in mirrored_joints[mirror_mask]:
                ax.add_patch(Circle((cx, cy), contact_radius, facecolor=mirror_contact_color, edgecolor="none", zorder=3))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        fig.savefig(path, format="pdf", transparent=True, bbox_inches=None, pad_inches=0)
    finally:
        plt.rcParams["path.simplify"] = old_path_simplify
        plt.close(fig)
    return path

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

    print(f"theta[0] = {fmt_float(theta[0] * 180 / np.pi)} deg, theta[-1] = {fmt_float(theta[-1] * 180 / np.pi)} deg")
    for i, (px, py) in enumerate(forces, start=1):
        mode = "con" if interfaces[i - 1].get("type") == "con" else "load"
        print(
            f"  joint {i:>2} [{mode}]: "
            f"Px={fmt_float(px)}, Py={fmt_float(py)}, "
            f"x={fmt_float(joints[i - 1, 0])}, y={fmt_float(joints[i - 1, 1])}"
        )

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
    max_force = np.max(np.linalg.norm(forces, axis=1))
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
