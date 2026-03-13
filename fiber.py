import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import newton, brentq
from scipy.integrate import quad


# ============================================================
# Parameters
# ============================================================
L_tot = 10.0
k_s = 1e5
k_b = 1e1

dt = 2e-5
gamma = 1e-2

n_steps = 800000
snap_every = 2000

# If you want only the origin fixed, keep only node 0 fixed.
# If you want the initial direction clamped too, also fix node 1.
clamp_direction = False


# ============================================================
# Parametric curve
# ============================================================
x = lambda t: 0.1 * t**3
y = lambda t: t
dx = lambda t: 0.3 * t**2
dy = lambda t: 1.0

# ============================================================
# Arc-length utilities
# ============================================================
def curve_length(x, y, l=10, dx=None, dy=None):
    def deriv(f, t):
        h = np.cbrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(t))
        return (f(t + h) - f(t - h)) / (2 * h)

    def v(t):
        dx_dt = deriv(x, t) if dx is None else dx(t)
        dy_dt = deriv(y, t) if dy is None else dy(t)
        return np.sqrt(dx_dt**2 + dy_dt**2)

    def F(t):
        return quad(v, 0, t)[0] - l

    # Better initial guess than x0=l for many curves
    t1 = newton(F, x0=max(1.0, l * 0.7), fprime=v)
    return t1


def parametric_lengths(x, y, t1, npts=101, l=10):
    m = npts - 1

    def chord(ta, tb):
        return np.hypot(x(tb) - x(ta), y(tb) - y(ta))

    def place_prefix(d):
        ts = [0.0]
        for _ in range(m - 1):
            ta = ts[-1]
            if chord(ta, t1) < d:
                raise ValueError("Chord length exceeds remaining curve length.")
            f = lambda tb: chord(ta, tb) - d
            ts.append(brentq(f, ta, t1))
        return np.array(ts)

    def residuals(d):
        ts = place_prefix(d)
        return chord(ts[-1], t1) - d

    d_lo = 1e-12
    d_hi = (l / m) * 1.001

    d = brentq(residuals, d_lo, d_hi)
    ts = place_prefix(d)
    ts = np.append(ts, t1)
    return ts, d


# ============================================================
# Build initial rod
# ============================================================
t1 = curve_length(x, y, L_tot, dx=dx, dy=dy)
ts, d = parametric_lengths(x, y, t1, npts=101, l=L_tot)

R0 = np.column_stack([x(ts), y(ts)])
N = len(R0)

l0 = np.linalg.norm(np.diff(R0, axis=0), axis=1)
h = np.mean(l0)

# Scaled effective stiffnesses
ks = k_s / h
kb = k_b / h**3

# Lumped masses
m = np.ones(N) * h
m[0] *= 0.5
m[-1] *= 0.5

fixed = np.zeros(N, dtype=bool)
fixed[0] = True
if clamp_direction:
    fixed[1] = True

R_fixed = R0.copy()


# ============================================================
# Forces and energies
# ============================================================
def compute_forces(R, V, l0, ks, kb, gamma=0.0, fixed=None):
    F = np.zeros_like(R)

    # Stretching
    D = R[1:] - R[:-1]
    S = np.linalg.norm(D, axis=1)
    eps_len = 1e-12
    S_safe = np.maximum(S, eps_len)

    U = D / S_safe[:, None]
    Fs = ks * (S - l0)[:, None] * U

    # Optional spring force clipping
    Fmax_seg = 1e6
    fmag = np.linalg.norm(Fs, axis=1)
    mask = fmag > Fmax_seg
    if np.any(mask):
        Fs[mask] *= (Fmax_seg / fmag[mask])[:, None]

    F[:-1] += Fs
    F[1:]  -= Fs

    # Bending
    Q = R[2:] - 2.0 * R[1:-1] + R[:-2]
    Fb = kb * Q

    # Optional bending clipping
    Bmax = 1e6
    bmag = np.linalg.norm(Fb, axis=1)
    mask = bmag > Bmax
    if np.any(mask):
        Fb[mask] *= (Bmax / bmag[mask])[:, None]

    F[:-2]  += -Fb
    F[1:-1] +=  2.0 * Fb
    F[2:]   += -Fb

    F += -gamma * V

    if fixed is not None:
        F[fixed] = 0.0

    return F


def energies(R, l0, ks, kb):
    D = R[1:] - R[:-1]
    S = np.linalg.norm(D, axis=1)
    E_stretch = 0.5 * ks * np.sum((S - l0)**2)

    Q = R[2:] - 2.0 * R[1:-1] + R[:-2]
    E_bend = 0.5 * kb * np.sum(np.sum(Q**2, axis=1))

    return E_stretch, E_bend, E_stretch + E_bend


# ============================================================
# Time stepping
# ============================================================
def verlet_step(R, V, m, dt, l0, ks, kb, gamma, fixed=None, R_fixed=None):
    F = compute_forces(R, V, l0, ks, kb, gamma, fixed)
    A = F / m[:, None]

    V_half = V + 0.5 * dt * A

    max_step = 0.02 * h
    max_v = max_step / dt

    vnorm = np.linalg.norm(V_half, axis=1)
    mask = vnorm > max_v
    if np.any(mask):
        V_half[mask] *= (max_v / vnorm[mask])[:, None]

    R_new = R + dt * V_half

    if fixed is not None and R_fixed is not None:
        R_new[fixed] = R_fixed[fixed]

    if not np.all(np.isfinite(R_new)):
        raise FloatingPointError("Non-finite R_new in verlet_step_safe")

    F_new = compute_forces(R_new, V_half, l0, ks, kb, gamma, fixed)
    A_new = F_new / m[:, None]

    V_new = V_half + 0.5 * dt * A_new

    if fixed is not None:
        V_new[fixed] = 0.0

    if not np.all(np.isfinite(V_new)):
        raise FloatingPointError("Non-finite V_new in verlet_step_safe")

    return R_new, V_new


# ============================================================
# Main simulation loop
# ============================================================
R = R0.copy()
V = np.zeros_like(R0)

snapshots = []
energy_hist = []
step_list = []

for step in range(n_steps):
    R, V = verlet_step(
        R, V, m, dt, l0, ks, kb, gamma,
        fixed=fixed, R_fixed=R_fixed
    )

    if step % snap_every == 0:
        Es, Eb, Et = energies(R, l0, ks, kb)
        vmax = np.max(np.linalg.norm(V, axis=1))

        snapshots.append(R.copy())
        energy_hist.append((Es, Eb, Et))
        step_list.append(step)

        print(
            f"step={step:6d}, ({step/n_steps*100:.2f}%) | "
            f"Es={Es:.6e}, Eb={Eb:.6e}, Et={Et:.6e}, vmax={vmax:.3e}"
        )

        if vmax < 1e-10:
            print(f"Converged at step {step}")
            break


# ============================================================
# Static comparison plot
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(R0[:, 0], R0[:, 1], "--", label="initial")
ax.plot(R[:, 0], R[:, 1], label="verlet")
ax.scatter(R_fixed[fixed, 0], R_fixed[fixed, 1], color="red", label="fixed")
ax.set_title("Velocity Verlet final")
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()


# ============================================================
# Animation
# ============================================================
all_x = [R0[:, 0]]
all_y = [R0[:, 1]]

for S in snapshots:
    all_x.append(S[:, 0])
    all_y.append(S[:, 1])

xmin = min(np.min(a) for a in all_x)
xmax = max(np.max(a) for a in all_x)
ymin = min(np.min(a) for a in all_y)
ymax = max(np.max(a) for a in all_y)

pad_x = 0.05 * (xmax - xmin + 1e-12)
pad_y = 0.05 * (ymax - ymin + 1e-12)

fig_anim, ax_anim = plt.subplots(figsize=(7, 7))

all_x = [R0[:, 0]] + [S[:, 0] for S in snapshots]
all_y = [R0[:, 1]] + [S[:, 1] for S in snapshots]

xmin = min(np.min(a) for a in all_x)
xmax = max(np.max(a) for a in all_x)
ymin = min(np.min(a) for a in all_y)
ymax = max(np.max(a) for a in all_y)

pad_x = 0.05 * (xmax - xmin + 1e-12)
pad_y = 0.05 * (ymax - ymin + 1e-12)

ax_anim.set_xlim(xmin - pad_x, xmax + pad_x)
ax_anim.set_ylim(ymin - pad_y, ymax + pad_y)
ax_anim.set_aspect("equal", adjustable="box")
ax_anim.grid(True)
ax_anim.set_title("Velocity Verlet")

line_init, = ax_anim.plot(R0[:, 0], R0[:, 1], "--", alpha=0.35)
line, = ax_anim.plot([], [], lw=2)
ax_anim.scatter(R_fixed[fixed, 0], R_fixed[fixed, 1], color="red")

txt = ax_anim.text(0.02, 0.98, "", transform=ax_anim.transAxes, va="top")

def update(frame):
    Ri = snapshots[frame]
    line.set_data(Ri[:, 0], Ri[:, 1])

    Es, Eb, Et = energy_hist[frame]
    step = step_list[frame]

    txt.set_text(
        f"step = {step}\n"
        f"Es = {Es:.3e}\n"
        f"Eb = {Eb:.3e}\n"
        f"Et = {Et:.3e}"
    )

    return line, txt

ani = FuncAnimation(
    fig_anim,
    update,
    frames=len(snapshots),
    interval=50,
    blit=False,
    repeat=True
)

ani.save("fiber_verlet.gif", writer="pillow", fps=30)
plt.tight_layout()
plt.show()

"""
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-5, 5)
ax.set_ylim(0, L_tot + 2)
ax.set_aspect('equal')
plt.tight_layout()
fig.canvas.draw()


line, = ax.plot( y(t), x(t), 'k-', solid_capstyle='butt')

def update_lw(data_width=2.0):
    bbox = ax.get_window_extent()
    xlim = ax.get_xlim()
    lw = data_width * (bbox.width / (xlim[1] - xlim[0])) * 72 / fig.dpi
    line.set_linewidth(lw)
    fig.canvas.draw_idle()

update_lw()
fig.canvas.mpl_connect('resize_event', lambda e: update_lw())
ax.callbacks.connect('xlim_changed', lambda ax: update_lw())

#circle = Circle(C, r_C, edgecolor='r', facecolor='r', linewidth=2, zorder=5)
#ax.add_patch(circle)


plt.show()


"""

