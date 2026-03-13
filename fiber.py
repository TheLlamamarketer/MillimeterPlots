import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from scipy.optimize import newton, brentq
from scipy.integrate import quad
from numba import njit, prange
from matplotlib.animation import FFMpegWriter



# ============================================================
# Parameters
# ============================================================
L_tot = 0.1 

diam = 300e-6
rho = 2200.0
A = np.pi * (diam / 2)**2
I = np.pi * (diam / 2)**4 / 4

E = 73e9
k_s = E * A
k_b = E * I

n_steps = 1_000_000
snap_every = 1_000

n_points = 101

# If you want only the origin fixed, keep only node 0 fixed.
# If you want the initial direction clamped too, also fix node 1.
clamp_direction = False


# ============================================================
# Parametric curve
# ============================================================
L_ref = 10.0
shape_scale = L_tot / L_ref

# Keep the same curve shape as the L_tot=10 setup, scaled uniformly with L_tot.
x = lambda t: shape_scale * np.sin(0.3 * (t / shape_scale) ** 2)
y = lambda t: t
dx = lambda t: 0.6 * (t / shape_scale) * np.cos(0.3 * (t / shape_scale) ** 2)
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
        return quad(v, 0, t, limit=200)[0] - l

    # Scale initial guess with target length; avoids oversized guesses for small l.
    x0 = max(1e-8, l * 0.8)

    try:
        t1 = newton(F, x0=x0, fprime=v, maxiter=100)
    except RuntimeError:
        # Fallback: bracket and solve robustly if Newton stalls.
        t_hi = max(1e-8, l)
        while F(t_hi) < 0.0:
            t_hi *= 2.0
            if t_hi > 1e6:
                raise RuntimeError("Failed to bracket arc-length root in curve_length")
        t1 = brentq(F, 0.0, t_hi)

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
ts, d = parametric_lengths(x, y, t1, npts=n_points, l=L_tot)

R0 = np.column_stack([x(ts), y(ts)])
N = len(R0)

l0 = np.linalg.norm(np.diff(R0, axis=0), axis=1)
h = np.mean(l0)

# Scaled effective stiffnesses
ks = k_s / h
kb = k_b / h**3

c_ax = np.sqrt(E / rho)
c_b  = np.sqrt(E * I / (rho * A))

dt_ax = 0.2 * h / c_ax
dt_b  = 0.05 * h * h / (np.pi**2 * c_b)

dt = min(dt_ax, dt_b)

zeta = 2
omega1 = (1.875104068711961**2) * c_b / L_tot**2
eta = 2 * zeta * omega1

# Lumped masses
m = np.ones(N) * rho * A * h
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
@njit(cache=True) 
def compute_forces(R, V, l0, ks, kb, eta, m, fixed=None):
    F = np.zeros_like(R)
    N = R.shape[0]

    # Stretching
    for i in range(N - 1):
        dx = R[i + 1, 0] - R[i, 0]
        dy = R[i + 1, 1] - R[i, 1]
        s = np.sqrt(dx*dx + dy*dy)
        if s < 1e-12:
            s = 1e-12 
        fac = ks * (s - l0[i]) / s
        fx = fac * dx
        fy = fac * dy
        
        F[i, 0] += fx
        F[i, 1] += fy
        F[i + 1, 0] -= fx
        F[i + 1, 1] -= fy
    
    for i in range(N - 2):
        qx = R[i + 2, 0] - 2.0 * R[i + 1, 0] + R[i, 0]
        qy = R[i + 2, 1] - 2.0 * R[i + 1, 1] + R[i, 1]
        
        fx = kb * qx
        fy = kb * qy
        
        F[i, 0] += -fx
        F[i, 1] += -fy
        F[i + 1, 0] += 2.0 * fx
        F[i + 1, 1] += 2.0 * fy
        F[i + 2, 0] += -fx
        F[i + 2, 1] += -fy
    
    for i in range(N):
        F[i, 0] -= eta * m[i] * V[i, 0]
        F[i, 1] -= eta * m[i] * V[i, 1]

        if fixed[i]:
            F[i, 0] = 0.0
            F[i, 1] = 0.0

    return F

@njit(cache=True, parallel=True)
def energies(R, l0, ks, kb):
    N = R.shape[0]
    Es = 0.0
    Eb = 0.0
    for i in prange(N - 1):
        dx = R[i + 1, 0] - R[i, 0]
        dy = R[i + 1, 1] - R[i, 1]
        s = np.sqrt(dx*dx + dy*dy)
        if s < 1e-12:
            s = 1e-12 
            
        Es += 0.5 * ks * (s - l0[i])*(s - l0[i])
        
    for i in prange(N - 2):
        qx = R[i + 2, 0] - 2.0 * R[i + 1, 0] + R[i, 0]
        qy = R[i + 2, 1] - 2.0 * R[i + 1, 1] + R[i, 1]
        
        Eb += 0.5 * kb * (qx**2 + qy**2)

    return Es, Eb, Es + Eb


# ============================================================
# Time stepping
# ============================================================
@njit(cache=True)
def verlet_simulation(R0, m, dt, l0, ks, kb, fixed, R_fixed, n_steps, snap_every, h, eta):
    R = R0.copy()
    N = R0.shape[0]
    V = np.zeros_like(R0)

    n_steps_i = int(n_steps)
    snap_every_i = int(snap_every)
    
    max_snaps = n_steps_i // snap_every_i + 1
    snapshots = np.empty((max_snaps, N, 2), dtype=R0.dtype)
    energy_hist = np.empty((max_snaps, 3), dtype=R0.dtype)
    step_list = np.empty(max_snaps, dtype=np.int64)
    
    ns = 0
    
    for step in range(n_steps_i):
        vmax = 0.0

        F = compute_forces(R, V, l0, ks, kb, eta, m, fixed)
        
        for i in range(N):
            V[i, 0] += 0.5 * dt * F[i, 0] / m[i]
            V[i, 1] += 0.5 * dt * F[i, 1] / m[i]
        
        for i in range(N):
            R[i, 0] += dt * V[i, 0]
            R[i, 1] += dt * V[i, 1]
            if fixed[i]:
                R[i, 0] = R_fixed[i, 0]
                R[i, 1] = R_fixed[i, 1]


        F = compute_forces(R, V, l0, ks, kb, eta, m, fixed)

        for i in range(N):
            V[i, 0] += 0.5 * dt * F[i, 0] / m[i]
            V[i, 1] += 0.5 * dt * F[i, 1] / m[i]
            
            if fixed[i]:
                V[i, 0] = 0.0
                V[i, 1] = 0.0

            vn = np.sqrt(V[i, 0]*V[i, 0] + V[i, 1]*V[i, 1])
            if vn > vmax:
                vmax = vn

        if step % snap_every_i == 0:
            Es, Eb, Et = energies(R, l0, ks, kb)
            snapshots[ns] = R.copy()
            energy_hist[ns, 0] = Es
            energy_hist[ns, 1] = Eb
            energy_hist[ns, 2] = Et
            step_list[ns] = step
            ns += 1
            
            print(step / n_steps_i * 100, "%")
            
            if vmax < 1e-15 and step > 1000:
                print(f"Converged at step {step}")
                break

    return R, V, snapshots[:ns], energy_hist[:ns], step_list[:ns]


# ============================================================
# Main simulation loop
# ============================================================
R = R0.copy()
V = np.zeros_like(R0)

snapshots = []
energy_hist = []
step_list = []

R, V, snapshots, energy_hist, step_list = verlet_simulation(
    R0, m, dt, l0, ks, kb, fixed, R_fixed, n_steps, snap_every, h, eta
)


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


# ===============================================================
# Energy plot
# ===============================================================

energy_hist = np.array(energy_hist)
fig_energy, ax_energy = plt.subplots(figsize=(10, 5))
ax_energy.plot(energy_hist[:, 0], label="Stretching")
ax_energy.plot(energy_hist[:, 1], label="Bending")
ax_energy.plot(energy_hist[:, 2], label="Total")
ax_energy.set_xlabel("Step")
ax_energy.set_ylabel("Energy")
ax_energy.legend()
ax_energy.grid(True)

# ============================================================
# Animation
# ============================================================
all_x = [R0[:, 0]]
all_y = [R0[:, 1]]

tot_len = np.sum(np.linalg.norm(np.diff(snapshots, axis=1), axis=2), axis=1)

for S in snapshots:
    all_x.append(S[:, 0])
    all_y.append(S[:, 1])

xmin = min(np.min(a) for a in all_x)
xmax = max(np.max(a) for a in all_x)
ymin = min(np.min(a) for a in all_y)
ymax = max(np.max(a) for a in all_y)

pad_x = 0.05 * (xmax - xmin + 1e-12)
pad_y = 0.05 * (ymax - ymin + 1e-12)

fig_anim, ax_anim = plt.subplots(figsize=(8.5, 7))
fig_anim.subplots_adjust(right=0.78)
info_ax = fig_anim.add_axes([0.80, 0.12, 0.18, 0.76])
info_ax.axis("off")

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
line,      = ax_anim.plot([], [], lw=2, animated=True)
ax_anim.scatter(R_fixed[fixed, 0], R_fixed[fixed, 1], color="red")

txt = info_ax.text(
    0.0,
    1.0,
    "",
    transform=info_ax.transAxes,
    va="top",
    ha="left",
    family="monospace",
    animated=True,
)

def init():
    line.set_data([], [])
    txt.set_text("")
    return line, txt

def update(frame):
    Ri = snapshots[frame]
    line.set_data(Ri[:, 0], Ri[:, 1])

    Es, Eb, Et = energy_hist[frame]
    step = step_list[frame]

    txt.set_text(
        f"step = {step}\n"
        f"L = {tot_len[frame]:.3f}\n"
        f"Es = {Es:.3e}\n"
        f"Eb = {Eb:.3e}\n"
        f"Et = {Et:.3e}"
    )
    return line, txt

ani = FuncAnimation(
    fig_anim,
    update,
    frames=len(snapshots),
    init_func=init,
    interval=40,
    blit=True,
    repeat=True,
    cache_frame_data=False
)

fig_anim.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))

# Resolve ffmpeg path robustly, including common Windows layouts.
ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
if not ffmpeg_path:
    env_override = os.environ.get("FFMPEG_PATH")
    if env_override and os.path.isfile(env_override):
        ffmpeg_path = env_override

if not ffmpeg_path and os.name == "nt":
    fallback_dirs = [
        r"C:\Tools\ffmpeg-8.0.1-essentials_build\bin",
    ]
    for folder in fallback_dirs:
        candidate = os.path.join(folder, "ffmpeg.exe")
        if os.path.isfile(candidate):
            ffmpeg_path = candidate
            break

if ffmpeg_path:
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path

if animation.writers.is_available("ffmpeg"):
    writer = FFMpegWriter(fps=20, bitrate=1800)
    ani.save(
        "fiber_verlet.mp4",
        writer=writer,
        dpi=90
    )
    print(f"Saved animation to fiber_verlet.mp4 using {ffmpeg_path}")
else:
    print("ffmpeg not found; skipping animation export (MP4 only)")

plt.show()