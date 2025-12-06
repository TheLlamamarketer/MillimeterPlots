import numpy as np
import pyvista as pv

# ======================
# PARAMETERS
# ======================
A1 = 0.5 * 1.0
A2 = 0.5 * 0.5
b  = 0.4                   # spatial / temporal frequency scaling
theta0 = np.pi / 2         # base phase difference

handedness_1 = 1           # first wave
handedness_2 = -1          # second wave

loops_per_run = 2          # how many times the wave cycles over full z
vary_theta    = True       # whether to vary theta over time

frames = 120
fps    = 30

mode   = "circular"        # 'linear' or 'circular'
lineup = "side_view"       # 'normal', 'side_view', 'front_view'
end_z  = 10.0

name = (
    "Linear_Polarization"
    if mode == "linear"
    else f"Circular_Polarization_{lineup}"
)

# ======================
# FIELD DEFINITION
# ======================
def field_components(z, t, A, phase_offset, handedness):
    """
    Return Ex(z,t), Ey(z,t) for given amplitude, phase offset and handedness.
    z : array
    t : scalar (frame time)
    """
    phase = 2 * np.pi * (b * z + t) + phase_offset
    Ex = A * np.sin(phase)
    Ey = handedness * A * np.cos(phase)
    return Ex, Ey

# grids
n_z = 1000
z = np.linspace(0, end_z, n_z)
phi = np.linspace(0, 2 * np.pi, 400)

# ======================
# INITIAL DATA (frame 0)
# ======================
t0 = 0.0
theta_init = 0.0 if vary_theta else theta0

Ex1_0, Ey1_0 = field_components(z, t0, A1, 0,          handedness_1)
Ex2_0, Ey2_0 = field_components(z, t0, A2, theta_init, handedness_2)

if mode == "linear":
    Ey1_0 = np.zeros_like(Ey1_0)
    Ex2_0 = np.zeros_like(Ex2_0)

Ex_res_0 = Ex1_0 + Ex2_0
Ey_res_0 = Ey1_0 + Ey2_0

# ellipse at z=0
Ex1_ell_0 = A1 * np.sin(phi)
Ey1_ell_0 = handedness_1 * A1 * np.cos(phi)
Ex2_ell_0 = A2 * np.sin(phi + theta_init)
Ey2_ell_0 = handedness_2 * A2 * np.cos(phi + theta_init)
Ex_ell_0  = Ex1_ell_0 + Ex2_ell_0
Ey_ell_0  = Ey1_ell_0 + Ey2_ell_0
z_ell_0   = np.zeros_like(Ex_ell_0)

# ======================
# PLOTTER SETUP
# ======================
pv.set_plot_theme("document")  # light theme, decent defaults

pl = pv.Plotter(window_size=(900, 650))
pl.set_background("white")

# simple origin axes (small triad)
pl.add_axes_at_origin(x_color="black", y_color="black", z_color="black")

# camera / view presets
if lineup == "normal":
    # angled view
    pl.camera_position = [
        (3.0, 4.0, 14.0),      # camera position
        (0.0, 0.0, end_z / 2), # focal point
        (0.0, 1.0, 0.0),       # up
    ]
elif lineup == "side_view":
    # look along +x so you see Ey–z plane
    pl.camera_position = "yz"
elif lineup == "front_view":
    # look along +z, so you see Ex–Ey plane
    pl.camera_position = "xy"

# ======================
# CREATE MESHES
# ======================
def make_line_mesh(x, y, zcoords):
    pts = np.column_stack((x, y, zcoords))
    return pv.lines_from_points(pts)

# waves as tubes
wave1_mesh   = make_line_mesh(Ex1_0,   Ey1_0,   z)
wave2_mesh   = make_line_mesh(Ex2_0,   Ey2_0,   z)
wave_res_mesh = make_line_mesh(Ex_res_0, Ey_res_0, z)

wave1_act = pl.add_mesh(
    wave1_mesh,
    color=(0.25, 0.4, 1.0),
    line_width=4,
    smooth_shading=True,
)
wave2_act = pl.add_mesh(
    wave2_mesh,
    color=(1.0, 0.2, 0.2),
    line_width=4,
    smooth_shading=True,
)
wave_res_act = pl.add_mesh(
    wave_res_mesh,
    color=(0.0, 0.6, 0.0),
    line_width=6,
    smooth_shading=True,
)

# ellipse at z=0
ellipse_mesh = make_line_mesh(Ex_ell_0, Ey_ell_0, z_ell_0)
ellipse_act = pl.add_mesh(
    ellipse_mesh,
    color=(0.5, 0.5, 0.5),
    line_width=3,
    smooth_shading=True,
)

# initial arrows (we'll recreate them per frame)
arrow1_act = None
arrow2_act = None
arrow_res_act = None

# ======================
# GIF OUTPUT
# ======================
pl.open_gif(f"{name}_pv.gif")

# ======================
# ANIMATION LOOP
# ======================
for frame in range(frames):
    # fast time: how many wave cycles we see
    t = frame / float(frames) * loops_per_run

    # slow parameter: theta change
    if vary_theta:
        progress = frame / float(frames - 1 if frames > 1 else 1)
        theta_now = 2 * np.pi * progress   # 0 → 2π over the animation
    else:
        theta_now = theta0

    # recompute field components
    Ex1, Ey1 = field_components(z, t, A1, 0,          handedness_1)
    Ex2, Ey2 = field_components(z, t, A2, theta_now,  handedness_2)

    if mode == "linear":
        Ey1 = np.zeros_like(Ey1)
        Ex2 = np.zeros_like(Ex2)

    Ex_res = Ex1 + Ex2
    Ey_res = Ey1 + Ey2

    # update line meshes (just move the points)
    wave1_mesh.points = np.column_stack((Ex1,     Ey1,     z))
    wave2_mesh.points = np.column_stack((Ex2,     Ey2,     z))
    wave_res_mesh.points = np.column_stack((Ex_res, Ey_res, z))

    # update ellipse at z=0
    Ex1_ell = A1 * np.sin(phi)
    Ey1_ell = handedness_1 * A1 * np.cos(phi)
    Ex2_ell = A2 * np.sin(phi + theta_now)
    Ey2_ell = handedness_2 * A2 * np.cos(phi + theta_now)
    Ex_ell  = Ex1_ell + Ex2_ell
    Ey_ell  = Ey1_ell + Ey2_ell
    z_ell   = np.zeros_like(Ex_ell)

    ellipse_mesh.points = np.column_stack((Ex_ell, Ey_ell, z_ell))

    # instantaneous field at z=0 for arrows
    Ex10, Ey10 = field_components(0.0, t, A1, 0,          handedness_1)
    Ex20, Ey20 = field_components(0.0, t, A2, theta_now,  handedness_2)

    if mode == "linear":
        Ey10 = 0.0
        Ex20 = 0.0

    Ex0_res = Ex10 + Ex20
    Ey0_res = Ey10 + Ey20

    # remove previous arrows
    for act in (arrow1_act, arrow2_act, arrow_res_act):
        if act is not None:
            pl.remove_actor(act)

    # create new arrow meshes (scale='auto' uses vector length)
    arrow1_mesh = pv.Arrow(start=(0, 0, 0),
                           direction=(Ex10, Ey10, 0),
                           scale='auto')
    arrow2_mesh = pv.Arrow(start=(0, 0, 0),
                           direction=(Ex20, Ey20, 0),
                           scale='auto')
    arrow_res_mesh = pv.Arrow(start=(0, 0, 0),
                              direction=(Ex0_res, Ey0_res, 0),
                              scale='auto')

    arrow1_act = pl.add_mesh(
        arrow1_mesh,
        color=(0.25, 0.4, 1.0),
        smooth_shading=True,
    )
    arrow2_act = pl.add_mesh(
        arrow2_mesh,
        color=(1.0, 0.2, 0.2),
        smooth_shading=True,
    )
    arrow_res_act = pl.add_mesh(
        arrow_res_mesh,
        color=(0.0, 0.6, 0.0),
        smooth_shading=True,
    )

    # write this frame to GIF
    pl.write_frame()

# finish and close
pl.close()
