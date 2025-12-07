import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

def load_wsxm_top(path, header_size=446):
    """
    Load a WSxM .top file. This routine attempts to auto-detect the header
    length and whether the binary data are float32 or float64. It falls back
    to the provided `header_size` when detection fails.
    """
    # read entire file bytes so we can probe header sizes/dtypes
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
        # check float64
        if remaining == expected_count * 8:
            chosen_header_size = hs
            chosen_dtype = "<f8"
            header_text = text
            found = True
            break
        # check float32
        if remaining == expected_count * 4:
            chosen_header_size = hs
            chosen_dtype = "<f4"
            header_text = text
            found = True
            break

    if not found:
        # fallback: use provided header_size and assume float64; try to
        # reshape by trimming or padding if necessary
        try:
            header_text = data_bytes[:header_size].decode("latin-1")
            meta = parse_meta_from_text(header_text)
        except Exception:
            meta = {}
        chosen_header_size = header_size
        chosen_dtype = "<f8"

    # parse meta from chosen header text (ensure keys present)
    if header_text is None:
        header_text = data_bytes[:chosen_header_size].decode("latin-1", errors="ignore")
    meta = parse_meta_from_text(header_text)

    rows = meta.get("rows")
    cols = meta.get("cols")
    if rows is None or cols is None:
        raise ValueError("Could not parse rows/cols from .top header")

    # extract binary payload and convert
    payload = data_bytes[chosen_header_size:]
    arr = np.frombuffer(payload, dtype=chosen_dtype)

    expected_size = rows * cols
    if arr.size < expected_size:
        # If too short, try switching dtype as last resort
        alt_dtype = "<f4" if chosen_dtype == "<f8" else "<f8"
        arr_alt = np.frombuffer(payload, dtype=alt_dtype)
        if arr_alt.size >= expected_size:
            arr = arr_alt
            chosen_dtype = alt_dtype

    if arr.size < expected_size:
        # As a final fallback, pad with NaNs so reshape won't fail
        padded = np.full(expected_size, np.nan, dtype=arr.dtype)
        padded[: arr.size] = arr
        arr = padded
    else:
        arr = arr[:expected_size]

    image = arr.reshape((rows, cols))
    image = image.copy()

    return image, meta, header_text


def logistic(x, z0, h, x0, k):
    return z0 + h * expit(k * (x - x0))

def fit_line_profiles(image,
                      min_points=10,
                      min_step_height_nm=None,
                      grad_sigma=2,
                      window_half_width=30,
                      max_k=5.0,
                      rms_rel_tol=0.3,
                      rms_abs_tol=0.2):
    
    
    """
        Fit each vertical line profile in the image to a logistic step function. 
        --
        
        image               : 2D array
        min_points         : minimum number of valid (non-NaN) points to attempt fit
        min_step_height_nm : minimum estimated step height to attempt fit
        grad_sigma         : Gaussian smoothing sigma for gradient estimation
        window_half_width  : half-width of fitting window around estimated step
        max_k              : maximum absolute value of logistic steepness parameter
        rms_rel_tol        : relative RMS tolerance for fit acceptance
        rms_abs_tol        : absolute RMS tolerance for fit acceptance
        
        Returns a list of fitted parameters per column, or None if fit failed.
        
    """
    
    
    rows, cols = image.shape
    row_idx = np.arange(rows)

    fit_params = []

    for col in range(cols):
        line_profile = image[:, col]

        # Mask NaNs
        valid_mask = ~np.isnan(line_profile)
        if valid_mask.sum() < min_points:
            fit_params.append(None)
            continue

        y_data = row_idx[valid_mask]
        z_data = line_profile[valid_mask]

        # Optional quick check: do we even have a significant step?
        if min_step_height_nm is not None:
            low  = np.nanpercentile(z_data, 5)
            high = np.nanpercentile(z_data, 95)
            h_est = high - low
            if h_est < min_step_height_nm:
                fit_params.append(None)
                continue

        # Smooth for robust gradient detection
        z_smooth = gaussian_filter1d(z_data, sigma=grad_sigma)
        grad = np.gradient(z_smooth)
        if grad.size == 0:
            fit_params.append(None)
            continue

        x0_init = y_data[np.argmax(np.abs(grad))]

        # Restrict to a window around the apparent step
        win_mask = (y_data >= x0_init - window_half_width) & (y_data <= x0_init + window_half_width)
        if win_mask.sum() < min_points:
            fit_params.append(None)
            continue

        y_fit = y_data[win_mask]
        z_fit = z_data[win_mask]

        # Initial guesses within the window
        z0_init = np.nanpercentile(z_fit, 5)
        h_init = np.nanpercentile(z_fit, 95) - z0_init
        if h_init == 0:
            fit_params.append(None)
            continue

        # Just give k a reasonable magnitude; let the fit decide the sign
        k_init = 0.2

        p0 = [z0_init, h_init, x0_init, k_init]

        lower = [-np.inf, -np.inf,      y_fit.min(), -max_k]
        upper = [ np.inf,  np.inf,      y_fit.max(),  max_k]

        try:
            popt, _ = curve_fit(logistic, y_fit, z_fit, p0=p0,
                                bounds=(lower, upper), maxfev=20000)
        except Exception:
            fit_params.append(None)
            continue

        # Quality check
        z_model = logistic(y_fit, *popt)
        
        res = z_fit - z_model
        # exclude large outliers from z_fit data
        outlier_mask = np.abs(res) < 3 * np.std(res)
        
        if outlier_mask.sum() < min_points:
            fit_params.append(None)
            continue
        else:
            try:
                popt, _ = curve_fit(logistic, y_fit[outlier_mask], z_fit[outlier_mask], p0=popt,
                                    bounds=(lower, upper), maxfev=20000)
            except Exception:
                fit_params.append(None)
                continue
            z_model = logistic(y_fit, *popt)
            
            res = z_fit - z_model
        rms = np.sqrt(np.mean((res)**2))

        if rms_rel_tol is None and rms_abs_tol is None:
            fit_params.append(popt)
        else:
            rel_limit = rms_rel_tol * abs(popt[1]) if rms_rel_tol is not None else 0.0
            abs_limit = rms_abs_tol if rms_abs_tol is not None else 0.0
            if rms > max(rel_limit, abs_limit):
                fit_params.append(None)
            else:
                fit_params.append(popt)

    return fit_params

top_file = "FP/STM/2ba.top"
image, meta, header = load_wsxm_top(top_file)

image[image == 0] = np.nan

x_size_nm = meta["x_amp"] * 1e9  # meters → nm
y_size_nm = meta["y_amp"] * 1e9
z_scale_nm = 1e9 # meters → nm
    
image *= z_scale_nm  # convert height map to nm units








# Determine common vmin/vmax for consistent colorbars
vmin = np.nanmin(image)
vmax = np.nanmax(image)

fig1 = plt.figure(1, figsize=(6, 5))
im1 = plt.imshow(image, origin="lower", cmap="viridis", extent=[0, x_size_nm, 0, y_size_nm], vmin=vmin, vmax=vmax)
cb1 = plt.colorbar(im1, label=f"Height (nm)")
plt.title("STM height map from 5fa.top")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.tight_layout()

fit_params = fit_line_profiles(image, window_half_width=10, min_step_height_nm=0.2, rms_rel_tol=0.5, rms_abs_tol=0.3, max_k=3.0)

fig2 = plt.figure(2, figsize=(12, 6))
# Build a fitted image by evaluating the fitted logistic per column
rows, cols = image.shape
fitted_image = np.full((rows, cols), np.nan)
for col, params in enumerate(fit_params):
    if params is None:
        continue
    try:
        fitted_image[:, col] = logistic(np.arange(rows), *params)
    except Exception:
        fitted_image[:, col] = np.nan

im2 = plt.imshow(fitted_image, origin="lower", cmap="viridis", extent=[0, x_size_nm, 0, y_size_nm],
                 vmin=vmin, vmax=vmax)
cb2 = plt.colorbar(im2, label=f"Fitted height (nm)")
plt.title("Fitted logistic per-column")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.tight_layout()


fig3 = plt.figure(3, figsize=(6, 5))
residual = image - fitted_image
v = np.nanmax(np.abs(residual))
im3 = plt.imshow(residual, origin="lower", cmap="seismic",
                 extent=[0, x_size_nm, 0, y_size_nm],
                 vmin=-v, vmax=v)
cb3 = plt.colorbar(im3, label=f"Height difference (nm)")
plt.title("STM height map residual (original - fitted)")
plt.xlabel("x (nm)")
plt.ylabel("y (nm)")
plt.tight_layout()
#plt.show()


fig4 = plt.figure(4, figsize=(8, 8))
valid_cols = []
while len(valid_cols) < min(4, cols):
    random_cols = np.random.choice(cols, size=min(4, cols), replace=False)
    for col in random_cols:
        if len(valid_cols) >= min(4, cols):
            break
        valid_mask = ~np.isnan(image[:, col])
        if np.any(valid_mask):
            valid_cols.append(col)

for idx, example_col in enumerate(valid_cols, 1):
    plt.subplot(2, 2, idx)
    valid_mask = ~np.isnan(image[:, example_col])
    y_vals = np.arange(rows)[valid_mask] * (y_size_nm / rows)
    z_vals = image[:, example_col][valid_mask]
    plt.plot(y_vals, z_vals, 'k.', label='data')
    if fit_params[example_col] is not None:
        plt.plot(np.arange(rows) * (y_size_nm / rows), fitted_image[:, example_col], 'r-', label='fit')
    plt.xlabel('y (nm)')
    plt.ylabel('height z (nm)')
    plt.title(f'Column {example_col} profile and fit')
    plt.legend()

plt.tight_layout()

# plot histogram of fitted step heights
heights = [abs(p[1]) for p in fit_params if p is not None]

fig6 = plt.figure(6, figsize=(6,4))
# scatter only at columns where a fit exists
x_positions = x_size_nm / cols * np.arange(cols)
heights_arr = np.array([abs(p[1]) if p is not None else np.nan for p in fit_params])
mask = ~np.isnan(heights_arr)
plt.scatter(x_positions[mask], heights_arr[mask], color='blue', alpha=0.7)
plt.xlabel("x position (nm)")
plt.ylabel("Fitted step height (nm)")

fig7 = plt.figure(7, figsize=(6,4))
plt.scatter(x_size_nm / cols * np.arange(cols), [p[2] * (y_size_nm / rows) if p is not None else np.nan for p in fit_params], color='green', alpha=0.7)
plt.xlabel("x position (nm)")
plt.ylabel("Fitted step center y0 (nm)")






fig5 = plt.figure(5, figsize=(6,4))
plt.hist(heights, bins=90, color='blue', alpha=0.7, density=True)

# Fit gaussian to heights
heights = np.array(heights)

med = np.median(heights)
mad = np.median(np.abs(heights - med))
mad_sigma = 1.4826 * mad   # MAD → σ for a normal

# keep values within 3σ of median
mask = np.abs(heights - med) < 3 * mad_sigma
h_good = heights[mask]

mean = h_good.mean()
std = h_good.std(ddof=1)   # unbiased sample std
stderr = std / np.sqrt(h_good.size)

print(f"Mean step height = {mean:.3f} ± {stderr:.3f} n, which corresponds to {mean / 0.335:.3f} ± {stderr / 0.335:.3f} layers of graphene (0.335 nm per layer)")

plt.hist(h_good, bins=40, color='red', alpha=0.7, density=True)

mu, sigma = norm.fit(h_good)
x = np.linspace(min(h_good), max(h_good), 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Gaussian fit\nμ={mu:.3f}, σ={sigma:.3f}')

plt.xlabel("Fitted step height (nm)")
plt.ylabel("Probability density")
plt.title("Histogram of Fitted Step Heights")
plt.legend()
plt.tight_layout()
print(f"Estimated average height not from gaussian fit: {abs(np.mean(heights)):.5g} nm ± {np.std(heights):.5g} nm")




# --- Robust pre-cleaning with MAD, as you already do ---
med = np.median(heights)
mad = np.median(np.abs(heights - med))
mad_sigma = 1.4826 * mad  # MAD → σ for a normal

mask = np.abs(heights - med) < 3 * mad_sigma
h_good = heights[mask]

print(f"Using {h_good.size} / {heights.size} points after 3·MAD cut")

# --- Simple mean/std estimate for comparison (single Gaussian) ---
mean_simple = h_good.mean()
std_simple  = h_good.std(ddof=1)
stderr_simple = std_simple / np.sqrt(h_good.size)

print(
    f"Single-Gaussian estimate: {mean_simple:.4f} ± {stderr_simple:.4f} nm "
    f"(std = {std_simple:.4f} nm)"
)

# --- Fit Gaussian mixture models with different numbers of components ---
def fit_best_gmm(data: np.ndarray, max_components: int = 5):
    X = data.reshape(-1, 1)
    models = []
    bics   = []
    for k in range(1, max_components + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=0,
            n_init=8
        )
        gmm.fit(X)
        models.append(gmm)
        bics.append(gmm.bic(X))
    bics = np.array(bics)
    best_idx = int(np.argmin(bics))
    best_k   = best_idx + 1
    best_gmm = models[best_idx]
    return best_gmm, best_k, bics

gmm, best_k, bics = fit_best_gmm(h_good, max_components=7)

print("BIC for K=1..7:", bics)
print(f"Selected number of Gaussians: K = {best_k}")



# --- Extract mixture parameters ---
means  = gmm.means_.ravel()               # component means
vars_  = gmm.covariances_.ravel()
sigmas = np.sqrt(vars_)
weights = gmm.weights_.ravel()

# Sort components by mean for nicer printing/plotting
order = np.argsort(means)
means  = means[order]
sigmas = sigmas[order]
weights = weights[order]

print("Components (mean ± sigma, weight):")
for i, (m, s, w) in enumerate(zip(means, sigmas, weights), start=1):
    print(f"  #{i}: μ = {m:.4f} nm, σ = {s:.4f} nm, weight = {w:.3f}")

# --- Try to estimate the fundamental step height d from component means ---

# Only use "significant" components (ignore tiny-weight junk)
weight_threshold = 0.05
sig_idx = weights > weight_threshold
means_sig = means[sig_idx]

d_est = np.nan
if means_sig.size >= 2:
    # Differences between adjacent means
    diffs = np.diff(means_sig)
    # Robust estimate: median of differences
    d_est = np.median(diffs)
    print(f"Estimated layer spacing from GMM: d ≈ {d_est:.4f} nm "
          f"(from {means_sig.size} significant components)")
else:
    print("Not enough significant components to robustly estimate d.")

# --- Plot histogram + mixture fit ---
fig8 = plt.figure(8, figsize=(6, 4))

# Original histogram of all cleaned data
plt.hist(h_good, bins=40, density=True, alpha=0.4, label="Data (cleaned)")

# x-range for PDF plotting
x = np.linspace(h_good.min(), h_good.max(), 500)

# Full mixture PDF
logprob = gmm.score_samples(x.reshape(-1, 1))
pdf_mix = np.exp(logprob)
plt.plot(x, pdf_mix, linewidth=2, label=f"GMM mixture (K={best_k})")

# Plot each component separately
for i, (m, s, w) in enumerate(zip(means, sigmas, weights), start=1):
    component_pdf = w * norm.pdf(x, m, s)
    plt.plot(x, component_pdf, linestyle='--', linewidth=1.5,
             label=f"Comp {i}: μ={m:.3f}, σ={s:.3f}")

plt.xlabel("Fitted step height (nm)")
plt.ylabel("Probability density")
plt.title("Histogram of Fitted Step Heights + Gaussian Mixture")
plt.legend(fontsize=8)
plt.tight_layout()

if not np.isnan(d_est):
    print(f"Estimated d / 0.335 nm ≈ {d_est / 0.335:.3f}")


# show every plot at the same time
plt.show()






