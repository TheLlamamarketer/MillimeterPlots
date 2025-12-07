import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm

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


def estimate_step_line(image, grad_sigma=2):
    rows, cols = image.shape
    y_idx = np.arange(rows)
    y_step = np.full(cols, np.nan)

    for col in range(cols):
        col_data = image[:, col]
        mask = ~np.isnan(col_data)
        if mask.sum() < 5:
            continue

        y = y_idx[mask]
        z = col_data[mask]

        z_smooth = gaussian_filter1d(z, sigma=grad_sigma)
        g = np.gradient(z_smooth)
        if g.size == 0:
            continue

        y_step[col] = y[np.argmax(np.abs(g))]

    # fit a line y_step(x) = a*x + b
    valid = ~np.isnan(y_step)
    x = np.arange(cols)[valid]
    y = y_step[valid]
    a, b = np.polyfit(x, y, 1)
    y_step_global = a * np.arange(cols) + b
    return y_step_global


def estimate_terrace_plane(image, y_step_global, band_halfwidth=3):
    rows, cols = image.shape
    y_idx, x_idx = np.indices((rows, cols))

    mask = ~np.isnan(image)

    # mask out a band around the step so it doesn't bias the plane
    for col in range(cols):
        y0 = y_step_global[col]
        if np.isnan(y0):
            continue
        band = (y_idx[:, col] >= y0 - band_halfwidth) & (y_idx[:, col] <= y0 + band_halfwidth)
        mask[band, col] = False

    x = x_idx[mask].ravel()
    y = y_idx[mask].ravel()
    z = image[mask].ravel()

    if z.size < 3:
        # fallback: no plane
        return 0.0, 0.0, 0.0

    A = np.c_[x, y, np.ones_like(x)]
    coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeff
    return a, b, c

def subtract_plane(image, a, b, c):
    rows, cols = image.shape
    y_idx, x_idx = np.indices((rows, cols))
    plane = a * x_idx + b * y_idx + c
    return image - plane, plane


def logistic(y, z0, h, x0, k, m_fixed):
    return m_fixed * y + z0 + h * expit(k * (y - x0))

def fit_line_profiles(image_flat,
                      y_step_global,
                      min_points=8,
                      window_half_width=10,
                      y0_margin=3,
                      rms_rel_tol=None,
                      rms_abs_tol=None):
    rows, cols = image_flat.shape
    row_idx = np.arange(rows)
    fit_params = []

    for col in range(cols):
        line_profile = image_flat[:, col]
        valid = ~np.isnan(line_profile)
        if valid.sum() < min_points:
            fit_params.append(None)
            continue

        y_data = row_idx[valid]
        z_data = line_profile[valid]

        y0_global = y_step_global[col]
        if np.isnan(y0_global):
            fit_params.append(None)
            continue

        # window around global step position
        win = (y_data >= y0_global - window_half_width) & (y_data <= y0_global + window_half_width)
        if win.sum() < min_points:
            fit_params.append(None)
            continue

        y_fit = y_data[win]
        z_fit = z_data[win]

        # initial guesses
        z0_init = np.nanpercentile(z_fit, 10)
        h_init  = np.nanpercentile(z_fit, 90) - z0_init
        if h_init <= 0:
            fit_params.append(None)
            continue

        k_init  = 0.2
        y0_init = y0_global
        p0 = [z0_init, h_init, y0_init, k_init]

        y0_min = y0_global - y0_margin
        y0_max = y0_global + y0_margin

        lower = [-np.inf, 0.0,    y0_min, -5.0]
        upper = [ np.inf, np.inf, y0_max,  5.0]

        try:
            popt, _ = curve_fit(logistic, y_fit, z_fit, p0=p0,
                                bounds=(lower, upper), maxfev=20000)
        except Exception:
            fit_params.append(None)
            continue

        z_model = logistic(y_fit, *popt)
        rms = np.sqrt(np.mean((z_fit - z_model)**2))

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



top_file = "FP/STM/5faa.top"
image, meta, header = load_wsxm_top(top_file)
image[image == 0] = np.nan
x_size_nm = meta["x_amp"] * 1e9
y_size_nm = meta["y_amp"] * 1e9
image *= 1e9  # z → nm

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



# 1) global step line
y_step_global = estimate_step_line(image, grad_sigma=2)

# 2) global plane from terraces
a, b, c = estimate_terrace_plane(image, y_step_global, band_halfwidth=3)
image_flat, plane = subtract_plane(image, a, b, c)

# 3) per-column pure logistic fits on flattened image
fit_params = fit_line_profiles(image_flat,
                               y_step_global,
                               min_points=4,
                               window_half_width=10,
                               y0_margin=3,
                               rms_rel_tol=None,   # start lenient
                               rms_abs_tol=None)

# 4) build fitted images
rows, cols = image.shape
y = np.arange(rows)
fitted_flat = np.full((rows, cols), np.nan)
for col, params in enumerate(fit_params):
    if params is None:
        continue
    z0, h, y0, k = params
    fitted_flat[:, col] = logistic(y, z0, h, y0, k)

fitted_image = fitted_flat + plane


fig2 = plt.figure(2, figsize=(12, 6))


rows, cols = image.shape
fitted_image = np.full((rows, cols), np.nan)
for col, params in enumerate(fit_params):
    if params is None:
        continue
    fitted_image[:, col] = logistic(np.arange(rows), *params)


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
heights = [p[1] for p in fit_params if p is not None]
fig5 = plt.figure(5, figsize=(6,4))
plt.hist(heights, bins=900, color='blue', alpha=0.7, density=True)

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



# show every plot at the same time
plt.show()

