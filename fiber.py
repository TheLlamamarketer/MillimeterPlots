import numpy as np
import os
import shutil
from numba import njit, prange
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import newton, brentq
from scipy.integrate import quad
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, Normalize


# ============================================================
# Parameters
# ============================================================
L_tot = 0.1 

diam = 330e-6
rho = 2200.0
A = np.pi * (diam / 2)**2
I = np.pi * (diam / 2)**4 / 4

E = 73e9
k_s = E * A
k_b = E * I
kc = 1e5

n_steps = 2_000_000
snap_every = 10_000

time_speedup = 1.0

n_points = 151

save_animation = True

# Tensile failure threshold used for diagnostics coloring.
# Typical pristine silica fibers can be several GPa; adjust for your material quality.
sigma_tensile_ult = 1.0e9  # Pa



 

# ============================================================
# Parametric curve
# ============================================================
L_ref = 10.0
shape_scale = L_tot / L_ref
x = lambda t: 0.5 * shape_scale * np.sin(t *np.pi/2)
y = lambda t: shape_scale * t
dx = lambda t: 0.5 * shape_scale * np.pi/2 * np.cos(t *np.pi/2)
dy = lambda t: shape_scale * 1.0

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
        try:
            ts = place_prefix(d)
        except ValueError:
            # d is too large for the curve's chord geometry; treat as overshoot
            return -d
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

# Lumped masses
m = np.ones(N) * rho * A * h
m[0] *= 0.5
m[-1] *= 0.5

# Scaled effective stiffnesses
ks = k_s / h
kb = k_b / h**3

c_ax = np.sqrt(E / rho)
c_b_wave = np.sqrt(E * I / (rho * A))

dt_ax = 0.2 * h / c_ax
dt_b  = 0.05 * h**2 / (np.pi**2 * c_b_wave)
dt_c = 0.05 * np.sqrt(np.min(m) / kc)

print(f"Time step limits: dt_ax={dt_ax:.3e}, dt_b={dt_b:.3e}, dt_c={dt_c:.3e}")

dt = time_speedup * min(dt_ax, dt_b, dt_c)


fixed = np.zeros(N, dtype=bool)
#fixed[0] = True

R_fixed = R0.copy()

# ============================================================
# Damping parameters
# ============================================================

# Global mass damping
zeta_global = 0.5
zeta_axial = 0.25
zeta_bend = 1.0

# Contact damping and friction
mu = 0.2
zeta_t = 1.0
mu_imp = 0.2

omega1 = (1.875104068711961**2) * c_b_wave / L_tot**2
eta = 2.0 * zeta_global * omega1

# Effective reduced masses for dashpot estimates
m_seg = 0.5 * np.mean(m)
m_bend = np.mean(m)

# Axial Kelvin-Voigt damping coefficient
c_s_damp = 2.0 * zeta_axial * np.sqrt(ks * m_seg)

# Bending-rate damping coefficient
c_b_damp = 2.0 * zeta_bend * np.sqrt(kb * m_bend)


# =============================================================
# Collision points
# =============================================================

def zeta_calc(e):
    if e <= 0.0:
        return 1.0
    if e >= 1.0:
        return 0.0
    le = abs(np.log(e))
    return le / np.sqrt(np.pi**2 + le**2)

zeta_c = zeta_calc(0.1)  # e = 1 is elastic collision, e = 0 is perfectly inelastic

P_rad = 0.003

P = np.array([[0, 0.01, P_rad], [0, 0.03, P_rad], [0, 0.05, P_rad], [0, 0.07, P_rad], [0, 0.09, P_rad]])

P_eff = P.copy()
P_eff[:, 2] += diam / 2


# ============================================================
# Forces and energies
# ============================================================
@njit(cache=True)
def compute_forces(R, V, l0, ks, kb, kc,
                   zeta_c, zeta_t, mu,
                   eta, c_s_damp, c_b_damp,
                   m, P, fixed):
    F = np.zeros_like(R)
    N = R.shape[0]

    # Contact with circular obstacles
    for i in range(N):
        for j in range(P.shape[0]):
            rx = R[i, 0] - P[j, 0]
            ry = R[i, 1] - P[j, 1]
            d = np.sqrt(rx * rx + ry * ry)

            delta = P[j, 2] - d
            if delta > 0.0:
                if d > 1e-12:
                    nx = rx / d
                    ny = ry / d
                else:
                    nx = 0.0
                    ny = 1.0

                tx = -ny
                ty = nx

                vn = V[i, 0] * nx + V[i, 1] * ny
                vt = V[i, 0] * tx + V[i, 1] * ty

                cc = 2.0 * zeta_c * np.sqrt(kc * m[i])
                ct = 2.0 * zeta_t * np.sqrt(kc * m[i])

                fn = kc * delta
                if vn < 0.0:
                    fn -= cc * vn

                if fn < 0.0:
                    fn = 0.0

                ft_trial = -ct * vt
                ft_lim = mu * fn

                if ft_trial > ft_lim:
                    ft = ft_lim
                elif ft_trial < -ft_lim:
                    ft = -ft_lim
                else:
                    ft = ft_trial

                F[i, 0] += fn * nx + ft * tx
                F[i, 1] += fn * ny + ft * ty

    # Stretching + axial internal damping
    for i in range(N - 1):
        dx = R[i + 1, 0] - R[i, 0]
        dy = R[i + 1, 1] - R[i, 1]
        s = np.sqrt(dx * dx + dy * dy)
        if s < 1e-12:
            s = 1e-12

        ux = dx / s
        uy = dy / s

        # Elastic stretching force
        fs = ks * (s - l0[i])

        # Axial relative speed
        dvx = V[i + 1, 0] - V[i, 0]
        dvy = V[i + 1, 1] - V[i, 1]
        vrel = dvx * ux + dvy * uy

        # Kelvin-Voigt dashpot
        fd = c_s_damp * vrel

        fx = (fs + fd) * ux
        fy = (fs + fd) * uy

        F[i, 0] += fx
        F[i, 1] += fy
        F[i + 1, 0] -= fx
        F[i + 1, 1] -= fy

    # Bending + bending-rate damping
    for i in range(N - 2):
        qx = R[i + 2, 0] - 2.0 * R[i + 1, 0] + R[i, 0]
        qy = R[i + 2, 1] - 2.0 * R[i + 1, 1] + R[i, 1]

        qvx = V[i + 2, 0] - 2.0 * V[i + 1, 0] + V[i, 0]
        qvy = V[i + 2, 1] - 2.0 * V[i + 1, 1] + V[i, 1]

        fx = kb * qx + c_b_damp * qvx
        fy = kb * qy + c_b_damp * qvy

        F[i, 0] += -fx
        F[i, 1] += -fy
        F[i + 1, 0] += 2.0 * fx
        F[i + 1, 1] += 2.0 * fy
        F[i + 2, 0] += -fx
        F[i + 2, 1] += -fy
    
    # gravity
    g = 9.81
    for i in range(N):
        F[i, 1] -= m[i] * g

    # Global damping
    for i in range(N):
        F[i, 0] -= eta * m[i] * V[i, 0]
        F[i, 1] -= eta * m[i] * V[i, 1]

        if fixed[i]:
            F[i, 0] = 0.0
            F[i, 1] = 0.0

    return F

@njit(cache=True)
def energies(R, l0, ks, kb, V, m):
    N = R.shape[0]
    Es = 0.0
    Eb = 0.0
    Ek = 0.0

    for i in range(N - 1):
        dx = R[i + 1, 0] - R[i, 0]
        dy = R[i + 1, 1] - R[i, 1]
        s = np.sqrt(dx*dx + dy*dy)
        if s < 1e-12:
            s = 1e-12 
            
        Es += 0.5 * ks * (s - l0[i])*(s - l0[i])
        
    for i in range(N - 2):
        qx = R[i + 2, 0] - 2.0 * R[i + 1, 0] + R[i, 0]
        qy = R[i + 2, 1] - 2.0 * R[i + 1, 1] + R[i, 1]
        
        Eb += 0.5 * kb * (qx**2 + qy**2)
    
    for i in range(N):
        Ek += 0.5 * m[i] * (V[i, 0]**2 + V[i, 1]**2)

    # With hard contact projection, obstacle collisions do not store elastic
    # penetration energy. The conservative potential here is rod strain energy.
    Ep = Es + Eb
    return Es, Eb, Ek, Ep




@njit(cache=True)
def resolve_node_circle_contacts(R_old, R, V, P, fixed, m, mu_imp):
    N = R.shape[0]
    eps = 1e-14

    for i in range(N):
        if fixed[i]:
            continue

        x0 = R_old[i, 0]
        y0 = R_old[i, 1]
        x1 = R[i, 0]
        y1 = R[i, 1]

        dx = x1 - x0
        dy = y1 - y0

        best_s = 2.0
        best_j = -1

        # Swept node-circle intersection
        a = dx * dx + dy * dy
        if a > eps:
            for j in range(P.shape[0]):
                cx = P[j, 0]
                cy = P[j, 1]
                rad = P[j, 2]

                ox = x0 - cx
                oy = y0 - cy

                b = 2.0 * (ox * dx + oy * dy)
                c = ox * ox + oy * oy - rad * rad

                disc = b * b - 4.0 * a * c
                if disc >= 0.0:
                    sq = np.sqrt(disc)
                    s1 = (-b - sq) / (2.0 * a)

                    if 0.0 <= s1 <= 1.0 and s1 < best_s:
                        best_s = s1
                        best_j = j

        # Move node to first contact point
        if best_j >= 0:
            cx = P[best_j, 0]
            cy = P[best_j, 1]
            rad = P[best_j, 2]

            hx = x0 + best_s * dx
            hy = y0 + best_s * dy

            rx = hx - cx
            ry = hy - cy
            d = np.sqrt(rx * rx + ry * ry)

            if d > eps:
                nx = rx / d
                ny = ry / d
            else:
                nx = 0.0
                ny = 1.0

            tx = -ny
            ty = nx

            R[i, 0] = cx + (rad + 1e-12) * nx
            R[i, 1] = cy + (rad + 1e-12) * ny

            vn = V[i, 0] * nx + V[i, 1] * ny
            vt = V[i, 0] * tx + V[i, 1] * ty

            Jn = 0.0
            if vn < 0.0:
                Jn = m[i] * (-vn)
                V[i, 0] -= vn * nx
                V[i, 1] -= vn * ny

            Jt_trial = -m[i] * vt
            Jt_lim = mu_imp * Jn

            if Jt_trial > Jt_lim:
                Jt = Jt_lim
            elif Jt_trial < -Jt_lim:
                Jt = -Jt_lim
            else:
                Jt = Jt_trial

            V[i, 0] += (Jt / m[i]) * tx
            V[i, 1] += (Jt / m[i]) * ty

        # Safety projection
        for j in range(P.shape[0]):
            cx = P[j, 0]
            cy = P[j, 1]
            rad = P[j, 2]

            rx = R[i, 0] - cx
            ry = R[i, 1] - cy
            d = np.sqrt(rx * rx + ry * ry)

            if d < rad:
                if d > eps:
                    nx = rx / d
                    ny = ry / d
                else:
                    nx = 0.0
                    ny = 1.0

                tx = -ny
                ty = nx

                R[i, 0] = cx + (rad + 1e-12) * nx
                R[i, 1] = cy + (rad + 1e-12) * ny

                vn = V[i, 0] * nx + V[i, 1] * ny
                vt = V[i, 0] * tx + V[i, 1] * ty

                Jn = 0.0
                if vn < 0.0:
                    Jn = m[i] * (-vn)
                    V[i, 0] -= vn * nx
                    V[i, 1] -= vn * ny

                Jt_trial = -m[i] * vt
                Jt_lim = mu_imp * Jn

                if Jt_trial > Jt_lim:
                    Jt = Jt_lim
                elif Jt_trial < -Jt_lim:
                    Jt = -Jt_lim
                else:
                    Jt = Jt_trial

                V[i, 0] += (Jt / m[i]) * tx
                V[i, 1] += (Jt / m[i]) * ty

    return R, V



# ============================================================
# Time stepping
# ============================================================
@njit(cache=True)
def _verlet_chunk(R, V, l0, ks, kb, kc,
                  zeta_c, zeta_t, mu,
                  eta, c_s_damp, c_b_damp,
                  m, fixed, R_fixed, dt, snap_every, step_offset,
                  n_chunk, max_snaps, snapshots, energy_hist, step_list, force_list, ns,
                  P, quiet_count, v_tol, min_steps_for_conv, mu_imp, quiet_needed=5):

    N = R.shape[0]
    converged = False
    steps_done = 0

    for local_step in range(n_chunk):
        step = step_offset + local_step
        steps_done = local_step + 1
        vmax = 0.0

        F = compute_forces(
            R, V, l0, ks, kb, kc,
            zeta_c, zeta_t, mu,
            eta, c_s_damp, c_b_damp,
            m, P, fixed
        )

        for i in range(N):
            V[i, 0] += 0.5 * dt * F[i, 0] / m[i]
            V[i, 1] += 0.5 * dt * F[i, 1] / m[i]

        R_old = R.copy()

        for i in range(N):
            R[i, 0] += dt * V[i, 0]
            R[i, 1] += dt * V[i, 1]
            if fixed[i]:
                R[i, 0] = R_fixed[i, 0]
                R[i, 1] = R_fixed[i, 1]

        R, V = resolve_node_circle_contacts(R_old, R, V, P, fixed, m, mu_imp)

        F = compute_forces(
            R, V, l0, ks, kb, kc,
            zeta_c, zeta_t, mu,
            eta, c_s_damp, c_b_damp,
            m, P, fixed
        )

        for i in range(N):
            V[i, 0] += 0.5 * dt * F[i, 0] / m[i]
            V[i, 1] += 0.5 * dt * F[i, 1] / m[i]

            if fixed[i]:
                V[i, 0] = 0.0
                V[i, 1] = 0.0

            vmag = np.sqrt(V[i, 0] * V[i, 0] + V[i, 1] * V[i, 1])
            if vmag > vmax:
                vmax = vmag

        if step % snap_every == 0:
            if ns < max_snaps:
                Es, Eb, Ek, Ep = energies(R, l0, ks, kb, V, m)
                snapshots[ns] = R
                energy_hist[ns, 0] = Es
                energy_hist[ns, 1] = Eb
                energy_hist[ns, 2] = Ek
                energy_hist[ns, 3] = Ep
                step_list[ns] = step
                force_list[ns] = F
                ns += 1

            if step >= min_steps_for_conv and vmax < v_tol:
                quiet_count += 1
            else:
                quiet_count = 0

            if quiet_count >= quiet_needed:
                converged = True
                break

    return R, V, ns, converged, quiet_count, steps_done


def verlet_simulation(R0, m, dt, l0, ks, kb, fixed, R_fixed,
                      n_steps, snap_every, h,
                      eta, c_s_damp, c_b_damp,
                      P, kc, zeta_c, zeta_t, mu, mu_imp,
                      chunk_size=50_000):

    R = R0.copy()
    N = R0.shape[0]
    V = np.zeros_like(R0)

    n_steps = int(n_steps)
    snap_every = int(snap_every)
    max_snaps = n_steps // snap_every + 1

    snapshots = np.empty((max_snaps, N, 2), dtype=R0.dtype)
    energy_hist = np.empty((max_snaps, 4), dtype=R0.dtype)
    step_list = np.empty(max_snaps, dtype=np.int64)
    force_list = np.empty((max_snaps, N, 2), dtype=R0.dtype)

    v_tol = 1e-2 * h / max(snap_every * dt, 1e-30)

    with tqdm(total=n_steps, desc="Simulating", unit="steps") as pbar:
        quiet_count = 0
        step_offset = 0
        ns = 0

        while step_offset < n_steps:
            n_chunk = min(chunk_size, n_steps - step_offset)

            R, V, ns, converged, quiet_count, steps_done = _verlet_chunk(
                R, V, l0, ks, kb, kc,
                zeta_c, zeta_t, mu,
                eta, c_s_damp, c_b_damp,
                m, fixed, R_fixed, dt, snap_every, step_offset,
                n_chunk, max_snaps, snapshots, energy_hist, step_list, force_list, ns,
                P, quiet_count, v_tol, 5 * snap_every, mu_imp
            )

            pbar.update(steps_done)
            step_offset += steps_done

            if converged:
                break

    return R, V, snapshots[:ns], energy_hist[:ns], step_list[:ns], force_list[:ns]


# ============================================================
# Main simulation loop
# ============================================================
R, V, snapshots, energy_hist, step_list, force_list = verlet_simulation(
    R0, m, dt, l0, ks, kb, fixed, R_fixed,
    n_steps, snap_every, h,
    eta, c_s_damp, c_b_damp,
    P_eff, kc, zeta_c, zeta_t, mu, mu_imp
)

force_node_mag = np.linalg.norm(force_list, axis=2)
force_segment_mag = 0.5 * (force_node_mag[:, :-1] + force_node_mag[:, 1:])
max_force_mag = float(force_segment_mag.max()) if force_segment_mag.size else 0.0


def tensile_failure_index_segments(R, l0, E, diam, sigma_ult):
    """Return per-segment failure index sigma_tension/sigma_ult."""
    n = R.shape[0]
    if n < 3:
        return np.zeros(max(n - 1, 0), dtype=R.dtype)

    # Axial strain per segment.
    dR = R[1:] - R[:-1]
    s = np.sqrt(dR[:, 0]**2 + dR[:, 1]**2)
    s = np.maximum(s, 1e-15)
    eps_ax = (s - l0) / np.maximum(l0, 1e-15)

    # Curvature estimate at nodes: kappa ~ turning angle / average adjacent arc length.
    seg = dR / s[:, None]
    dot = np.sum(seg[:-1] * seg[1:], axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    ds_mid = 0.5 * (s[:-1] + s[1:])
    kappa_mid = theta / np.maximum(ds_mid, 1e-15)

    kappa_node = np.zeros(n, dtype=R.dtype)
    kappa_node[1:-1] = kappa_mid
    kappa_node[0] = kappa_mid[0]
    kappa_node[-1] = kappa_mid[-1]
    kappa_seg = 0.5 * (kappa_node[:-1] + kappa_node[1:])

    # Outer-fiber tensile strain on the tensile side.
    eps_b = 0.5 * diam * kappa_seg
    eps_tension = np.maximum(eps_ax + eps_b, 0.0)

    sigma_tension = E * eps_tension
    return sigma_tension / max(sigma_ult, 1e-15)


failure_index = np.empty((snapshots.shape[0], snapshots.shape[1] - 1), dtype=snapshots.dtype)
for k in range(snapshots.shape[0]):
    failure_index[k] = tensile_failure_index_segments(snapshots[k], l0, E, diam, sigma_tensile_ult)

max_failure_index = float(failure_index.max()) if failure_index.size else 0.0
print(f"Max tensile failure index sigma/sigma_ult: {max_failure_index:.3f}")

norm = PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0, clip=False)



# ==============================================================
# Plotting prep
# ==============================================================


def ribbon_quads(R, thickness):
    """Build one quad per segment with thickness in data units."""
    half = 0.5 * thickness
    n = R.shape[0]

    tangents = np.zeros_like(R)
    tangents[1:-1] = R[2:] - R[:-2]
    tangents[0] = R[1] - R[0]
    tangents[-1] = R[-1] - R[-2]

    lengths = np.sqrt(tangents[:, 0]**2 + tangents[:, 1]**2)
    lengths = np.maximum(lengths, 1e-15)

    normals = np.empty_like(R)
    normals[:, 0] = -tangents[:, 1] / lengths
    normals[:, 1] = tangents[:, 0] / lengths

    left = R + half * normals
    right = R - half * normals

    quads = np.empty((n - 1, 4, 2), dtype=R.dtype)
    quads[:, 0, :] = left[:-1]
    quads[:, 1, :] = left[1:]
    quads[:, 2, :] = right[1:]
    quads[:, 3, :] = right[:-1]
    return quads

cmap = LinearSegmentedColormap.from_list(
    "risk_map",
    ["#1a9850", "#CBCB46", "#d73027"]
).copy()
cmap.set_over("purple")   # values > 1.0

# ============================================================
# Static comparison plot
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 7))

quads = ribbon_quads(R, diam)
fiber = PolyCollection(quads, cmap=cmap, norm=norm, edgecolors="none")
fiber.set_array(failure_index[-1])

ax.plot(R0[:, 0], R0[:, 1], "--", label="initial")
ax.add_collection(fiber)

#ax.plot(R[:, 0], R[:, 1], label="verlet")
ax.scatter(R_fixed[fixed, 0], R_fixed[fixed, 1], color="red", label="fixed")

for j in range(P.shape[0]):
    circle = plt.Circle((P[j, 0], P[j, 1]), P[j, 2], color="orange", alpha=0.5, label="obstacle" if j == 0 else None)
    ax.add_patch(circle)

colorbar = plt.colorbar(fiber, ax=ax, pad=0.01)
colorbar.set_label("Failure index $(\\sigma/\\sigma_{\\text{ult}})$", rotation=270, labelpad=15)

ax.set_title("Velocity Verlet final")
ax.set_aspect("equal", adjustable="box")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()


# ===============================================================
# Energy plot
# ===============================================================
fig_energy, ax_energy = plt.subplots(figsize=(10, 5))
time = (step_list[1:] + 1) * dt
ax_energy.plot(time, energy_hist[1:, 2], label="Kinetic")
ax_energy.plot(time, energy_hist[1:, 3], label="Potential")
ax_energy.set_xlabel("Time (s)")
ax_energy.set_ylabel("Energy")
ax_energy.set_yscale("log")
ax_energy.legend()
ax_energy.grid(True)

# ============================================================
# Animation
# ============================================================
tot_len = np.sum(np.linalg.norm(np.diff(snapshots, axis=1), axis=2), axis=1)

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

ax_anim.plot(R0[:, 0], R0[:, 1], "--", alpha=0.35)
colorbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_anim, pad=0.01)
colorbar.set_label("Failure index $(\\sigma/\\sigma_{\\text{ult}})$", rotation=270, labelpad=15)

for j in range(P.shape[0]):
    circle = plt.Circle((P[j, 0], P[j, 1]), P[j, 2], color="orange", alpha=0.5, label="obstacle" if j == 0 else None)
    ax_anim.add_patch(circle)

line = ax_anim.add_collection(PolyCollection([], cmap=cmap, norm=norm, edgecolors="none", animated=True))
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
    line.set_verts([])
    line.set_array(np.empty(0, dtype=R0.dtype))
    txt.set_text("")
    return line, txt

def update(frame):
    Ri = snapshots[frame]

    quads = ribbon_quads(Ri, diam)
    line.set_verts(quads)
    line.set_array(failure_index[frame])

    Es, Eb, Ek, Ep = energy_hist[frame]
    step = step_list[frame]
    t_elapsed = (step + 1) * dt
    
    if t_elapsed < 1e-3:
        t_str = f"{t_elapsed*1e6:.3f} µs"
    elif t_elapsed < 1.0:
        t_str = f"{t_elapsed*1e3:.3f} ms"
    else:
        t_str = f"{t_elapsed:.3f} s"


    txt.set_text(
        f"step = {step}\n"
        f"t = {t_str}\n"
        f"L = {tot_len[frame]*1e3:.3f} mm\n"
        f"FI_max = {failure_index[frame].max():.3f}\n"
        f"Es = {Es:.3e}J\n"
        f"Eb = {Eb:.3e}J\n"
        f"Ek = {Ek:.3e}J\n"
        f"Ep = {Ep:.3e}J"
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

if save_animation:
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