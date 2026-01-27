import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, maximum_filter, label, center_of_mass
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import SmoothBivariateSpline
from sklearn.mixture import GaussianMixture
from itertools import combinations

import sys 
sys.path.append(str(Path(__file__).parent.parent.parent))

from Functions.help import print_round_val
from scipy.ndimage import gaussian_filter1d



def load_wsxm_top(path, header_size=446):
    """
    Load a WSxM .top file. This routine attempts to auto-detect the header
    length and whether the binary data are float32 or float64. It falls back
    to the provided `header_size` when detection fails.

    Returns
    -------
    image : (rows, cols) float ndarray
    meta  : dict  (keys: 'rows', 'cols', 'x_amp', 'y_amp', 'z_amp' if available)
    header_text : str
    """

    with open(path, "rb") as f:
        data_bytes = f.read()

    total_len = len(data_bytes)

    def parse_meta_from_text(text):
        meta = {}
        for line in text.splitlines():
            if "Number of columns" in line:
                try:
                    meta["cols"] = int(line.split(":")[1])
                except Exception:
                    pass
            elif "Number of rows" in line:
                try:
                    meta["rows"] = int(line.split(":")[1])
                except Exception:
                    pass
            elif "X Amplitude" in line:
                try:
                    meta["x_amp"] = float(line.split(":")[1].split()[0])
                except Exception:
                    pass
            elif "Y Amplitude" in line:
                try:
                    meta["y_amp"] = float(line.split(":")[1].split()[0])
                except Exception:
                    pass
            elif "Z Amplitude" in line:
                try:
                    meta["z_amp"] = float(line.split(":")[1].split()[0])
                except Exception:
                    pass
        return meta

    found = False
    header_text = None
    chosen_header_size = header_size
    chosen_dtype = "<f8"

    # Try to auto-detect header size and dtype
    for hs in range(128, 4097):
        if hs >= total_len:
            break
        try:
            text = data_bytes[:hs].decode("latin-1")
        except Exception:
            continue
        meta = parse_meta_from_text(text)
        if not ("rows" in meta and "cols" in meta):
            continue

        rows = meta["rows"]
        cols = meta["cols"]
        expected_count = rows * cols

        remaining = total_len - hs

        # float64?
        if remaining == expected_count * 8:
            chosen_header_size = hs
            chosen_dtype = "<f8"
            header_text = text
            found = True
            break
        # float32?
        if remaining == expected_count * 4:
            chosen_header_size = hs
            chosen_dtype = "<f4"
            header_text = text
            found = True
            break

    if not found:
        # fallback
        try:
            header_text = data_bytes[:header_size].decode("latin-1")
            meta = parse_meta_from_text(header_text)
        except Exception:
            meta = {}
        chosen_header_size = header_size
        chosen_dtype = "<f8"

    if header_text is None:
        header_text = data_bytes[:chosen_header_size].decode("latin-1", errors="ignore")
    meta = parse_meta_from_text(header_text)

    rows = meta.get("rows")
    cols = meta.get("cols")
    if rows is None or cols is None:
        raise ValueError("Could not parse rows/cols from .top header")

    payload = data_bytes[chosen_header_size:]
    arr = np.frombuffer(payload, dtype=chosen_dtype)

    expected_size = rows * cols
    if arr.size < expected_size:
        # try alternate dtype
        alt_dtype = "<f4" if chosen_dtype == "<f8" else "<f8"
        arr_alt = np.frombuffer(payload, dtype=alt_dtype)
        if arr_alt.size >= expected_size:
            arr = arr_alt
            chosen_dtype = alt_dtype

    if arr.size < expected_size:
        padded = np.full(expected_size, np.nan, dtype=arr.dtype)
        padded[:arr.size] = arr
        arr = padded
    else:
        arr = arr[:expected_size]

    image = arr.reshape((rows, cols)).copy()
    return image, meta, header_text

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

fft_top = "FP/STM/fft.top"
image_fft, meta_fft, header_fft = load_wsxm_top(fft_top)
image_fft = image_fft.astype(float)

# Replace zeros with NaN to avoid false peaks
image_fft[image_fft == 0] = np.nan

# Find local maxima
peaks = (image_fft == maximum_filter(image_fft, size=30))
peaks &= (image_fft > np.nanpercentile(image_fft, 99.9))  # threshold to avoid noise

# Label connected components and get center of mass
lbl_fft, n_ff = label(peaks)
positions_fft = np.array(center_of_mass(image_fft, lbl_fft, index=np.arange(1, n_ff+1)))

rows_f, cols_f = meta_fft["rows"], meta_fft["cols"]
x_f = np.linspace(-0.5 * meta_fft["x_amp"] / 1e9, 0.5 * meta_fft["x_amp"] / 1e9, cols_f)
y_f = np.linspace(-0.5 * meta_fft["y_amp"] / 1e9, 0.5 * meta_fft["y_amp"] / 1e9, rows_f)

# filter out the central peak by radius
radii = np.sqrt((positions_fft[:, 1] - cols_f // 2)**2 + (positions_fft[:, 0] - rows_f // 2)**2)
positions_fft = positions_fft[radii > 5]
peak_x = x_f[0] + (positions_fft[:, 1]) * (x_f[1] - x_f[0])
peak_y = y_f[0] + (positions_fft[:, 0]) * (y_f[1] - y_f[0])

ang = (np.degrees(np.arctan2(peak_y, peak_x)) + 360) % 360
ordr = np.argsort(ang); ang = ang[ordr]
gaps = np.diff(np.r_[ang, ang[0] + 360])          # 6 gaps, should be ~60°
print("sorted angles [deg]:", np.round(ang, 2))
print("neighbor gaps [deg]:", np.round(gaps, 2))

peak_x, peak_y = peak_x[ordr], peak_y[ordr]

vals = []
for i in range(6):
    b1 = np.array([peak_x[i], peak_y[i]])
    b2 = np.array([peak_x[(i+1)%6], peak_y[(i+1)%6]])
    A = np.linalg.inv(np.column_stack([b1, b2])).T
    a1, a2 = A[:,0], A[:,1]
    a = np.linalg.norm(a1); b = np.linalg.norm(a2)
    gamma = np.degrees(np.arccos(np.dot(a1,a2)/(a*b))); gamma = min(gamma, 180-gamma)
    vals.append((a,b,gamma))

vals = np.array(vals)
for i,(a,b,g) in enumerate(vals[:3]):
    a1, a2 = vals[i,0], vals[(i+3)%6,0]
    b1, b2 = vals[i,1], vals[(i+3)%6,1]
    g1, g2 = vals[i,2], vals[(i+3)%6,2]
    print(f"a = ({print_round_val(np.mean([a1, a2]), np.std([a1, a2], ddof=1))})nm, \\quad  b = ({print_round_val(np.mean([b1, b2]), np.std([b1, b2], ddof=1))})nm,\\quad  \\gamma={print_round_val(np.mean([g1, g2]), np.std([g1, g2], ddof=1))}^\\circ")



print('-'*40)


fig_fft = plt.figure(figsize=(10, 10))
im_fft = plt.imshow(image_fft*1e12/2, origin="lower", cmap="inferno", 
                    extent=[x_f[0], x_f[-1], y_f[0], y_f[-1]])
plt.colorbar(im_fft, label="FFT magnitude (pm)", shrink=0.7)
plt.scatter(peak_x, peak_y, s=20, c="cyan", marker=".", label="Detected peaks")
plt.title("FFT of 49f")
plt.xlabel("$k_x (1/nm)$")
plt.ylabel("$k_y (1/nm)$")
plt.tight_layout()
plt.legend()
plt.savefig(Path(__file__).with_name("STM_fft_peaks.pdf"), dpi=300, transparent=True)
plt.show()

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


# labels already computed above with proper sorting, don't overwrite!

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
im1 = plt.imshow(image, origin="lower", cmap="viridis", extent=[0, x_size_nm, 0, y_size_nm])
cb1 = plt.colorbar(im1, label=f"Height (pm)")
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

# --- Average distances in the 3 crystallographic directions ---
print("\nAverage distances in the 3 crystallographic directions:")
direction_distances = []
for i in range(3):
    mask_dir = (labels == i)
    distances_in_dir = dist_flat[mask_dir]
    
    if len(distances_in_dir) > 0:
        mean_dist = distances_in_dir.mean()
        std_dist = distances_in_dir.std()
        stderr_dist = std_dist / np.sqrt(len(distances_in_dir))
        direction_distances.append((mean_dist, stderr_dist, std_dist, len(distances_in_dir)))
        
        mu, mu_err = angles_mean[i]
        print(f"Direction {i} (angle {mu:.2f}° ± {mu_err:.2f}°):")
        print(f"  Mean distance: {mean_dist:.4f} ± {stderr_dist:.4f} nm")
        print(f"  Std deviation: {std_dist:.4f} nm")
        print(f"  N bonds: {len(distances_in_dir)}")
    else:
        direction_distances.append((np.nan, np.nan, np.nan, 0))
        print(f"Direction {i}: No bonds found")

# Calculate expected graphene nearest-neighbor distance (0.142 nm)
# and compare with measurements
graphene_nn = 0.142  # nm
print(f"\nLiterature graphene nearest-neighbor distance: {graphene_nn} nm")
print("Comparison with measurements:")
for i, (mean_dist, stderr_dist, std_dist, n) in enumerate(direction_distances):
    if not np.isnan(mean_dist):
        ratio = mean_dist / graphene_nn
        print(f"Direction {i}: measured = {mean_dist:.3f} nm ± {stderr_dist:.3f} nm, ratio = {ratio:.3f}")

# --- Histogram of distances per direction ---
fig_dist = plt.figure(figsize=(10, 4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    mask_dir = (labels == i)
    distances_in_dir = dist_flat[mask_dir]
    
    if len(distances_in_dir) > 0:
        mean_dist, stderr_dist, std_dist, n = direction_distances[i]
        mu_angle, err_angle = angles_mean[i]
        
        plt.hist(distances_in_dir, bins=30, color=colors[i], alpha=0.7, edgecolor='black')
        plt.axvline(mean_dist, color='red', linestyle='--', linewidth=2, 
                   label=f'μ = {mean_dist:.4f} nm')
        plt.axvline(graphene_nn, color='green', linestyle=':', linewidth=2,
                   label=f'Graphene = {graphene_nn} nm')
        plt.xlabel('Distance (nm)')
        plt.ylabel('Frequency')
        plt.title(f'Direction {i} ({mu_angle:.1f}°)')
        plt.legend(fontsize=8)

plt.tight_layout()
plt.savefig(Path(__file__).with_name("STM_distances_by_direction.pdf"), dpi=300, transparent=True)

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



# ======================================================================
# 2D FFT analysis of the graphite lattice
# ======================================================================

# --- FFT modulus from Gwyddion (optional, for comparison / plotting) ---
image_fft, meta_fft, header_fft = load_wsxm_top("FP/STM/fft.top")
image_fft = image_fft.astype(float)
image_fft[image_fft == 0] = np.nan
if np.any(np.isfinite(image_fft)):
    image_fft -= np.nanmin(image_fft)
image_fft = np.nan_to_num(image_fft, nan=0.0)

# --- Real-space topo used for the FFT (this defines dx, dy!) ----------
topo_img, topo_meta, _ = load_wsxm_top("FP/STM/topo_0049.f.top")
topo_img = topo_img.astype(float)
topo_img -= np.nanmean(topo_img)
topo_img = np.nan_to_num(topo_img, nan=0.0)

rows_t, cols_t = topo_img.shape
x_size_nm_t = topo_meta["x_amp"] * 1e9
y_size_nm_t = topo_meta["y_amp"] * 1e9
dx_nm = x_size_nm_t / cols_t
dy_nm = y_size_nm_t / rows_t

# --- FFT in Python ----------------------------------------------------
fft2 = np.fft.fft2(topo_img)
fft2_shift = np.fft.fftshift(fft2)
fft_mod_python = np.abs(fft2_shift)

# frequency axes in cycles / nm (Nyquist-limited)
kx = np.fft.fftshift(np.fft.fftfreq(cols_t, d=dx_nm))
ky = np.fft.fftshift(np.fft.fftfreq(rows_t, d=dy_nm))

fft_modulus = fft_mod_python   # or: fft_modulus = image_fft

rows_f, cols_f = fft_modulus.shape
Y_idx, X_idx = np.indices((rows_f, cols_f))
cy, cx = rows_f // 2, cols_f // 2
r_pix = np.sqrt((X_idx - cx)**2 + (Y_idx - cy)**2)

# For plotting we map pixel indices to kx, ky using the topo’s dx, dy
kx_plot = (np.arange(cols_f) - cols_f // 2) / (cols_f * dx_nm)
ky_plot = (np.arange(rows_f) - rows_f // 2) / (rows_f * dy_nm)

fig_fft = plt.figure(figsize=(6, 5))
im_fft = plt.imshow(
    np.log10(fft_modulus + 1e-16),
    origin="lower",
    cmap="magma",
    extent=[kx_plot.min(), kx_plot.max(), ky_plot.min(), ky_plot.max()],
)
plt.colorbar(im_fft, label=r"$\log_{10} |F(k_x,k_y)|$")
plt.xlabel(r"$k_x$ (cycles / nm)")
plt.ylabel(r"$k_y$ (cycles / nm)")
plt.title("2D FFT modulus of STM image")
plt.tight_layout()


plt.show()





