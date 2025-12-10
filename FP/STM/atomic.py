import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, maximum_filter, label, center_of_mass
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import SmoothBivariateSpline
from sklearn.mixture import GaussianMixture
from itertools import combinations
from Functions.help import print_round_val



def load_wsxm_top(path, header_size=446):
    with open(path, "rb") as f:
        header_bytes = f.read(header_size)
        header = header_bytes.decode("latin-1")

        # parse a few useful metadata fields
        meta = {}
        for line in header.splitlines():
            if "Number of columns" in line:
                meta["cols"] = int(line.split(":")[1])
            elif "Number of rows" in line:
                meta["rows"] = int(line.split(":")[1])
            elif "X Amplitude" in line:
                meta["x_amp"] = float(line.split(":")[1].split()[0])
            elif "Y Amplitude" in line:
                meta["y_amp"] = float(line.split(":")[1].split()[0])
            elif "Z Amplitude" in line:
                meta["z_amp"] = float(line.split(":")[1].split()[0])

        # read the remaining binary data as float64
        data = np.fromfile(f, dtype="<f8")

    rows, cols = meta["rows"], meta["cols"]
    image = data.reshape((rows, cols))  # height map

    return image, meta, header

def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    x, y = xy
    g = np.exp(-(((x - x0)**2)/(2*sigma_x**2)
                 + ((y - y0)**2)/(2*sigma_y**2)))
    return offset + amplitude * g


def fit_gaussians(image, positions_init, window_size=5):
    """
    image          : 2D array
    positions_init : array of (x, y) from peak finding
    window_size    : half-width in pixels around each peak
    """
    fitted_params = []

    rows, cols = image.shape
    for (x0, y0) in positions_init:
        x0 = int(round(x0))
        y0 = int(round(y0))

        x_min = max(0, x0 - window_size)
        x_max = min(cols, x0 + window_size + 1)
        y_min = max(0, y0 - window_size)
        y_max = min(rows, y0 + window_size + 1)

        region = image[y_min:y_max, x_min:x_max]

        # coordinate grid in pixel units, local to the region
        X, Y = np.meshgrid(np.arange(x_min, x_max),
                           np.arange(y_min, y_max))

        # initial guesses:
        amp0    = region.max() - region.min()
        offset0 = region.min()
        x0_init = x0
        y0_init = y0
        sigma0  = 1.5  # in pixels; STM spots are usually 1–3 px wide

        p0 = [amp0, x0_init, y0_init, sigma0, sigma0, offset0]

        try:
            popt, _ = curve_fit(
                gaussian_2d,
                (X.ravel(), Y.ravel()),
                region.ravel(),
                p0=p0,
                maxfev=5000
            )
            # popt = [A, x0, y0, sx, sy, offset]
            fitted_params.append(popt)
        except RuntimeError:
            continue

    return np.array(fitted_params)




top_file = "FP/STM/blanko49f.top"
image, meta, header = load_wsxm_top(top_file)

# Extract physical dimensions
rows, cols = meta["rows"], meta["cols"]
x_size_nm = meta["x_amp"] * 1e9  # meters → nm
y_size_nm = meta["y_amp"] * 1e9
z_scale_pm = 1e12 # meters → pm

image *= z_scale_pm  # convert height map to pm units

# Create physical coordinate arrays
x_nm = np.linspace(0, x_size_nm, cols)
y_nm = np.linspace(0, y_size_nm, rows)

img_smooth = gaussian_filter(image.astype(float), sigma=1.0)

# local maxima in a small neighborhood
local_max = (img_smooth == maximum_filter(img_smooth, size=7))

# apply threshold to avoid noise peaks
thr = np.percentile(img_smooth, 84) 
mask = local_max & (img_smooth > thr)

lbl, n = label(mask)
print("found peaks:", n)

# get sub-pixel initial positions by center-of-mass
initial_positions = np.array(center_of_mass(img_smooth, lbl, index=np.arange(1, n+1)))
positions_init = np.stack([initial_positions[:,1],
                           initial_positions[:,0]], axis=1) 



fitted = fit_gaussians(image, positions_init, window_size=4)

# Convert fitted positions from pixels to nm
fitted_nm = fitted.copy()
fitted_nm[:, 1] = fitted[:, 1] * (x_size_nm / cols)  # x positions
fitted_nm[:, 2] = fitted[:, 2] * (y_size_nm / rows)  # y positions
# Sigma values also need conversion
fitted_nm[:, 3] = fitted[:, 3] * (x_size_nm / cols)  # sigma_x
fitted_nm[:, 4] = fitted[:, 4] * (y_size_nm / rows)  # sigma_y

# Create physical coordinate grids for plotting
X_nm, Y_nm = np.meshgrid(x_nm, y_nm)

# Plot 1: Original height map with fitted positions
fig1 = plt.figure(1, figsize=(6, 5))
im1 = plt.imshow(image, origin="lower", cmap="viridis", extent=[0, x_size_nm, 0, y_size_nm])
plt.scatter(fitted_nm[:,1], fitted_nm[:,2], s=20, c="purple", label="Fitted positions")
cb1 = plt.colorbar(im1, label=f"Height (pm)")
plt.title("STM height map from .top")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.tight_layout()



# Plot 2: Residual
fig2 = plt.figure(2, figsize=(6, 5))
# Sum all fitted Gaussians in physical units
X_px, Y_px = np.meshgrid(np.arange(cols), np.arange(rows))
reconstructed = np.zeros_like(image, dtype=float)
window_size = 4

for A, x0, y0, sx, sy, c in fitted:
    x0_int = int(round(x0))
    y0_int = int(round(y0))
    
    x_min = max(0, x0_int - window_size)
    x_max = min(cols, x0_int + window_size + 1)
    y_min = max(0, y0_int - window_size)
    y_max = min(rows, y0_int + window_size + 1)
    
    X_region = X_px[y_min:y_max, x_min:x_max]
    Y_region = Y_px[y_min:y_max, x_min:x_max]
    
    gaussian_region = gaussian_2d((X_region, Y_region), A, x0, y0, sx, sy, 0.0)
    reconstructed[y_min:y_max, x_min:x_max] += gaussian_region

x0 = fitted[:, 1]
y0 = fitted[:, 2]
c  = fitted[:, 5]

bg_spline = SmoothBivariateSpline(x0, y0, c, kx=2, ky=2, s=1)
background = bg_spline(np.arange(cols), np.arange(rows), grid=True).reshape(image.shape)

residual = image - reconstructed
residual_masked = np.ma.masked_invalid(residual)

im2 = plt.imshow(residual_masked, origin="lower", cmap="seismic", extent=[0, x_size_nm, 0, y_size_nm], vmax=-np.nanmin(residual), vmin=np.nanmin(residual))
cb2 = plt.colorbar(im2, label=f"Height difference (pm)")
plt.title("STM height map residual (original - fitted)")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.tight_layout()


fig11 = plt.figure(11, figsize=(6, 4))
plt.hist(fitted[:, 0], bins=50, color='purple', alpha=0.7)
plt.xlabel("Amplitude (pm)")
plt.ylabel("Frequency")
plt.title("Histogram of fitted amplitudes")
plt.tight_layout()



fig10 = plt.figure(10, figsize=(6, 5))
plt.plot(np.arange(cols), reconstructed[120, :], color='purple', label="Fitted positions")
plt.plot(np.arange(cols), image[120, :], color='black', alpha=0.5, label="Original data")
plt.title("Line cut comparison at y=120")
plt.xlabel("x (nm)")
plt.ylabel("z (pm)")
plt.title("Fitted atomic positions")
plt.legend()
plt.tight_layout()


A = fitted[:, 0]   
mean_height = A.mean()
std_height  = A.std()

print(f"Mean atom height above local ground ≈ {mean_height:.3g} ± {std_height:.3g} pm")


xy = fitted_nm[:, 1:3]   # nm units
dx_nm = x_size_nm / cols
dy_nm = y_size_nm / rows


nn = NearestNeighbors(n_neighbors=min(7, len(xy)), algorithm='auto').fit(xy)
distances, indices = nn.kneighbors(xy)

dist = distances[:, 1:]   # (N, k-1)
idx  = indices[:, 1:]

mean_nn_dist = dist.mean()
std_nn_dist  = dist.std()

cutoff = 1.05 * mean_nn_dist
mask   = dist < cutoff

# --- masked distances and corresponding angles ---
vecs = xy[idx] - xy[:, None, :]     # (N, k-1, 2)
dist_masked   = np.where(mask, dist, np.nan)
angles        = np.arctan2(vecs[..., 1], vecs[..., 0])
angles_deg    = np.degrees(np.mod(angles, np.pi))

# --- flatten with the same mask ---
valid = ~np.isnan(dist_masked)
angles_deg_flat = angles_deg[valid]
dist_flat       = dist_masked[valid]

# --- GMM on angles ---
X = angles_deg_flat.reshape(-1, 1)
gmm = GaussianMixture(n_components=3, random_state=0).fit(X)

means  = gmm.means_.flatten()
stds   = np.sqrt(gmm.covariances_.flatten())
stderr = stds / np.sqrt(gmm.weights_ * len(X))

order = np.argsort(means)
means = means[order]
stderr = stderr[order]

labels_raw = gmm.predict(X)
labels = np.zeros_like(labels_raw)
for new_label, old_label in enumerate(order):
    labels[labels_raw == old_label] = new_label


angles_mean = [(means[i], stderr[i]) for i in range(3)] 

print("GMM results:")
for i in range(3):
    mu, err = angles_mean[i]
    print(f"family {i}: α = {mu:.3f}° ± {err:.3f}°")


colors = ['tab:blue', 'tab:orange', 'tab:green']

plt.figure(figsize=(6,4))
plt.hist(angles_deg_flat, bins=250, range=(0, 180), color='tab:blue', alpha=0.7)
for i in range(3):
    mu, err = angles_mean[i]
    plt.axvline(mu, color='tab:orange', linestyle='--', label=f"μ = {mu:.2f}° ± {err:.2f}°")
plt.xlabel("Angle (degrees)")
plt.ylabel("Frequency")
plt.title("Histogram of nearest neighbor angles")
plt.tight_layout()


labels = gmm.predict(X)

plt.figure(figsize=(6,4))

for i in range(3):
    mask_c = (labels == i)
    plt.scatter(dist_flat[mask_c], angles_deg_flat[mask_c], s=5, label=f"family {i}")
plt.xlabel("Distance to nearest neighbor (nm)")
plt.ylabel("Angle (degrees)")
plt.title("Nearest neighbor distances and angles colored by GMM family")
plt.legend()
plt.tight_layout()


d = np.array([dist_flat[labels == i].mean() for i in range(3)])
d0, d1, d2 = d

phi0, err0 = angles_mean[0]
phi1, err1 = angles_mean[1]
phi2, err2 = angles_mean[2]

b0 = d0 * np.array([np.cos(np.deg2rad(phi0)),
                    np.sin(np.deg2rad(phi0))])
b1 = d1 * np.array([np.cos(np.deg2rad(phi1)),
                    np.sin(np.deg2rad(phi1))])
b2 = d2 * np.array([np.cos(np.deg2rad(phi2)),
                    np.sin(np.deg2rad(phi2))])

B1 = np.column_stack([b0, b1]) 

a = (d0 + d2) / 2
a0 = a * np.array([1, 0])
a1 = a * np.array([np.cos(np.deg2rad(60)),
                   np.sin(np.deg2rad(60))])
a2 = a * np.array([np.cos(np.deg2rad(120)),
                   np.sin(np.deg2rad(120))])

A1 = np.column_stack([a0, a1])

F1 = B1 @ np.linalg.inv(A1)      # distortion
T1 = np.linalg.inv(F1)           # unwarping transform


print("Unwarping matrix T1:")
print(T1)


xy_corr = (T1 @ xy.T).T



plt.figure(figsize=(6,4))
plt.scatter(fitted_nm[:,1], fitted_nm[:,2], s=20, c="purple", label="Fitted positions")
for i in range(dist.shape[1]):
    vx = vecs[:, i, 0]
    vy = vecs[:, i, 1]
    valid = ~np.isnan(vx) & ~np.isnan(vy)
    if not np.any(valid):
        continue
    plt.quiver(
        xy[valid, 0], xy[valid, 1],
        vx[valid], vy[valid],
        angles='xy', scale_units='xy', scale=1,
        width=0.002, color='tab:orange', alpha=0.5
    )
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.tight_layout()


plt.figure(figsize=(6,4))
plt.scatter(xy_corr[:,0], xy_corr[:,1], s=20, c="purple", label="Fitted positions")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.tight_layout()

# apply T1 to the original image

X_flat = X_px.ravel()
Y_flat = Y_px.ravel()

XY_corr_flat = (T1 @ np.vstack([X_flat, Y_flat])).T
X_corr = XY_corr_flat[:, 0].reshape(rows, cols)
Y_corr = XY_corr_flat[:, 1].reshape(rows, cols)


fig_corr = plt.figure(3, figsize=(6, 5))
im_corr = plt.imshow(image, origin="lower", cmap="viridis", extent=[X_corr.min(), X_corr.max(), Y_corr.min(), Y_corr.max()])
cb_corr = plt.colorbar(im_corr, label=f"Height (pm)")
plt.title("Corrected STM height map")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.tight_layout()



pairwise_diffs = []
for (i, (phi_i, err_i)), (j, (phi_j, err_j)) in combinations(enumerate(angles_mean), 2):
    diff = (phi_i - phi_j) % 180
    diff = min(diff, 180 - diff)  # Ensure the difference is within [0, 90]
    diff_err = np.sqrt(err_i**2 + err_j**2)
    pairwise_diffs.append((i, j, diff, diff_err))

print("\nPairwise angle differences:")
for i, j, diff, diff_err in pairwise_diffs:
    print(f"{i}-{j}: {diff:6.2f} ± {diff_err:5.2f} °")

angles_rad = np.deg2rad(np.array([angle for angle, err in angles_mean]))
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))


for i, (t, (phi, err)) in enumerate(zip(angles_rad, angles_mean)):
    print(f"  Angle {i}: ${print_round_val(phi, err)}$ °")
    color = colors[i % len(colors)]

    ax.plot([t, t], [0, 1], lw=2, color=color, label=f"Angle {i}")
    ax.plot([t + np.pi, t + np.pi], [0, 1], lw=2, color=color, alpha=0.5)
    
    ax.text(
        t, 1.05,
        f'$ {i}: {phi:1.2f}° $',
        ha="center",
        va="center",
        fontsize=9,
        rotation=0 if 80 <= np.rad2deg(t) <= 100 else (np.rad2deg(t) if np.rad2deg(t) <= 90 else np.rad2deg(t) - 180),
        rotation_mode="anchor",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7)
    )

    err_rad = np.deg2rad(err)
    ax.fill_betweenx(
        [0, 1],
        t - err_rad,
        t + err_rad,
        color=color, alpha=0.3
    )

base_r = 0.3

for k, (i, j, diff, diff_err) in enumerate(pairwise_diffs):
    print(f" $ Arc {i} \\arrowright {j}: {print_round_val(diff, diff_err)} ° $")
    t1 = angles_rad[i] % np.pi
    t2 = angles_rad[j] % np.pi
    

    raw = (t2 - t1 + np.pi) % (2*np.pi) - np.pi

    if raw >  np.pi/2:
        raw -= np.pi
    elif raw < -np.pi/2:
        raw += np.pi
        
    diff_rad = np.deg2rad(diff)
    d = np.sign(raw) * diff_rad

    t_start = t1
    t_end   = t1 + d

    arc_t = np.linspace(t_start, t_end, 100)
    r_arc = np.full_like(arc_t, base_r + 0.05 * k)

    ax.plot(arc_t, r_arc, linestyle="-", linewidth=1, color="tab:purple")
    ax.fill_between(arc_t, 0, r_arc, alpha=0.5, color="tab:purple")

    t_mid = (t_start + t_end) / 2
    r_mid = base_r + 0.05 * k
    ax.text(t_mid, r_mid + 0.02, f"{i} → {j}: {diff:.1f}°", ha="center", va="bottom", fontsize=8, bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7))

ax.set_yticklabels([])
ax.set_title("", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(Path(__file__).with_name("STM_angles.pdf"), dpi=300, transparent=True)
plt.show()





