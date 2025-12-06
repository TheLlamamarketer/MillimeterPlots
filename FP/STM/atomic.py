import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter, maximum_filter, label, center_of_mass
from sklearn.neighbors import NearestNeighbors


import matplotlib.pyplot as plt

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

        # read the remaining binary data as float64 (little endian)
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
            popt, pcov = curve_fit(
                gaussian_2d,
                (X.ravel(), Y.ravel()),
                region.ravel(),
                p0=p0,
                maxfev=5000
            )
            # popt = [A, x0, y0, sx, sy, offset]
            fitted_params.append(popt)
        except RuntimeError:
            # fit didn’t converge; skip this peak
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
neighborhood = 7  # roughly size of one atom in pixels
local_max = (img_smooth == maximum_filter(img_smooth, size=neighborhood))

# apply threshold to avoid noise peaks
thr = np.percentile(img_smooth, 85)  # keep top 15% peaks
mask = local_max & (img_smooth > thr)

# label connected components (each peak region)
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
plt.show()


# Plot 2: Residual
fig2 = plt.figure(2, figsize=(6, 5))
# Sum all fitted Gaussians in physical units
X_px, Y_px = np.meshgrid(np.arange(cols), np.arange(rows))
reconstructed_no_off = np.zeros_like(image, float)
for A, x0, y0, sx, sy, c in fitted:
    reconstructed_no_off += gaussian_2d((X_px, Y_px), A, x0, y0, sx, sy, 0.0)

# estimate global background as mean or median
background = np.median(image - reconstructed_no_off)

reconstructed = reconstructed_no_off + background
residual = image - reconstructed
im2 = plt.imshow(residual, origin="lower", cmap="seismic", extent=[0, x_size_nm, 0, y_size_nm], vmax=-residual.min(), vmin=residual.min())
cb2 = plt.colorbar(im2, label=f"Height difference (pm)")
plt.title("STM height map residual (original - fitted)")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.tight_layout()
plt.show()

A = fitted[:, 0]   
mean_height = A.mean()
std_height  = A.std()

print(f"Mean atom height above local ground ≈ {mean_height:.3g} ± {std_height:.3g} pm")


xy = fitted_nm[:, 1:3]   # nm units
dx_nm = x_size_nm / cols
dy_nm = y_size_nm / rows


nn = NearestNeighbors(n_neighbors=min(7, len(xy)), algorithm='auto').fit(xy)
distances, indices = nn.kneighbors(xy)

dist = distances[:, 1:]  # exclude self-distance at index 0
idx     = indices[:, 1:]

mean_nn_dist = dist.mean()
std_nn_dist  = dist.std()

# exclude vectors longer than 1.2 times the mean nearest-neighbor distance
cutoff = 1.3 * mean_nn_dist
mask = dist < cutoff

# Keep the original `dist` for statistics but create a masked view for
# downstream vector computations while preserving the (N, k) shape.
dist_masked = np.where(mask, dist, np.nan)

# Create a safe index array for advanced indexing. Replace invalid indices
# with a valid placeholder (0) to avoid selecting the last element (-1)
# when indexing; we'll overwrite the corresponding vectors with NaN below.
idx_safe = idx.copy()
idx_safe[~mask] = 0

# Compute vectors (shape (N, k, 2)) and then set invalid entries to NaN
vecs = xy[idx_safe] - xy[:, None, :]
vecs[~mask] = np.nan

# Compute angles but ignore NaNs when making distributions/plots
angles = np.arctan2(vecs[..., 1], vecs[..., 0])  # [-π, π]
angles_deg = np.degrees(np.mod(angles, np.pi))

plt.figure(figsize=(6,4))
# ignore NaNs when building the histogram
angles_deg_flat = angles_deg[~np.isnan(angles_deg)].ravel()
plt.hist(angles_deg_flat, bins=250, range=(0, 180), color='tab:blue', alpha=0.7)
plt.xlabel("Angle (degrees)")
plt.ylabel("Frequency")
plt.title("Histogram of nearest neighbor angles")
plt.tight_layout()
plt.show()

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
plt.show()



print(f"Nearest-neighbour distance ≈ {mean_nn_dist:.3f} ± {std_nn_dist:.3f} nm")
