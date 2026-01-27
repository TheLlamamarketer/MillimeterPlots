import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from itertools import combinations
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import glob
import os
import csv

# ======================================================================
# I/O: WSxM .top loader
# ======================================================================

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


# ======================================================================
# Model functions and fitting utilities
# ======================================================================

def logistic(x, z0, h, x0, k):
    return z0 + h * expit(k * (x - x0))


def fit_line_profiles(
    image,
    min_points=10,
    min_step_height_nm=None,
    grad_sigma=2,
    window_half_width=30,
    max_k=5.0,
    rms_rel_tol=0.3,
    rms_abs_tol=0.2,
    averaging_radius=0,
):
    """
    Fit each vertical line profile in the image to a logistic step function.

    Parameters
    ----------
    image : 2D ndarray
        Height map in nm.
    min_points : int
        Minimum number of valid (non-NaN) points to attempt fit.
    min_step_height_nm : float or None
        Minimum estimated step height to attempt fit.
    grad_sigma : float
        Gaussian smoothing sigma for gradient estimation.
    window_half_width : int
        Half-width of fitting window around estimated step.
    max_k : float
        Maximum absolute value of logistic steepness parameter.
    rms_rel_tol : float or None
        Relative RMS tolerance for fit acceptance (relative to |h|).
    rms_abs_tol : float or None
        Absolute RMS tolerance for fit acceptance.
    averaging_radius : int
        Radius (in columns) to average for each line profile (0 = no averaging).

    Returns
    -------
    fit_params : list
        List of (popt, perr) per column, or None if fit failed.
        popt = [z0, h, x0, k]
        perr = 1σ uncertainties from covariance (np.nan where undefined).
    """
    rows, cols = image.shape
    row_idx = np.arange(rows)

    fit_params = []

    for col in range(cols):
        # column averaging
        if averaging_radius > 0:
            col_start = max(0, col - averaging_radius)
            col_end = min(cols, col + averaging_radius + 1)
            line_profile = np.nanmean(image[:, col_start:col_end], axis=1)
        else:
            line_profile = image[:, col]

        valid_mask = ~np.isnan(line_profile)
        if valid_mask.sum() < min_points:
            fit_params.append(None)
            continue

        y_data = row_idx[valid_mask]
        z_data = line_profile[valid_mask]

        # quick check for significant step
        if min_step_height_nm is not None:
            low = np.nanpercentile(z_data, 5)
            high = np.nanpercentile(z_data, 95)
            h_est = high - low
            if h_est < min_step_height_nm:
                fit_params.append(None)
                continue

        # gradient-based initial x0
        z_smooth = gaussian_filter1d(z_data, sigma=grad_sigma)
        grad = np.gradient(z_smooth)
        if grad.size == 0:
            fit_params.append(None)
            continue

        x0_init = y_data[np.argmax(np.abs(grad))]

        # restrict window around step
        win_mask = (y_data >= x0_init - window_half_width) & (y_data <= x0_init + window_half_width)
        if win_mask.sum() < min_points:
            fit_params.append(None)
            continue

        y_fit = y_data[win_mask]
        z_fit = z_data[win_mask]

        # initial guesses
        z0_init = np.nanpercentile(z_fit, 5)
        h_init = np.nanpercentile(z_fit, 95) - z0_init
        if h_init == 0:
            fit_params.append(None)
            continue

        k_init = 0.2
        p0 = [z0_init, h_init, x0_init, k_init]

        lower = [-np.inf, -np.inf, y_fit.min(), -max_k]
        upper = [np.inf, np.inf,   y_fit.max(),  max_k]

        try:
            popt, _ = curve_fit(
                logistic, y_fit, z_fit, p0=p0,
                bounds=(lower, upper), maxfev=20000
            )
        except Exception:
            fit_params.append(None)
            continue

        # quality check and outlier removal
        z_model = logistic(y_fit, *popt)
        res = z_fit - z_model
        outlier_mask = np.abs(res) < 3 * np.std(res)

        if outlier_mask.sum() < min_points:
            fit_params.append(None)
            continue

        try:
            popt, pcov = curve_fit(
                logistic, y_fit[outlier_mask], z_fit[outlier_mask],
                p0=popt, bounds=(lower, upper), maxfev=20000
            )
            if pcov is None:
                perr = np.full_like(popt, np.nan, dtype=float)
            else:
                perr = np.sqrt(np.diag(pcov))
                perr[~np.isfinite(perr)] = np.nan
        except Exception:
            fit_params.append(None)
            continue

        z_model = logistic(y_fit, *popt)
        res = z_fit - z_model
        rms = np.sqrt(np.mean(res ** 2))

        if rms_rel_tol is None and rms_abs_tol is None:
            fit_params.append((popt, perr))
        else:
            rel_limit = rms_rel_tol * abs(popt[1]) if rms_rel_tol is not None else 0.0
            abs_limit = rms_abs_tol if rms_abs_tol is not None else 0.0
            if rms > max(rel_limit, abs_limit):
                fit_params.append(None)
            else:
                fit_params.append((popt, perr))

    return fit_params


def fit_plane(z, X, Y, mask):
    """Fit z ≈ a x + b y + c on masked pixels."""
    xs = X[mask].ravel()
    ys = Y[mask].ravel()
    zs = z[mask].ravel()
    G = np.c_[xs, ys, np.ones_like(xs)]
    a, b, c = np.linalg.lstsq(G, zs, rcond=None)[0]
    return a, b, c


def fit_best_gmm(data: np.ndarray, max_components: int = 7, min_distance: float = 0.2):
    """Fit GMM(1..max_components) and choose best by BIC, enforcing minimum distance between components."""
    X = data.reshape(-1, 1)
    models = []
    bics = []
    for k in range(1, max_components + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=0,
            n_init=8
        )
        gmm.fit(X)

        # Check minimum distance between component means
        means = gmm.means_.ravel()
        if k > 1:
            min_dist = np.min(np.diff(np.sort(means)))
            if min_dist < min_distance:
                continue  # Skip this model if components are too close

        models.append(gmm)
        bics.append(gmm.bic(X))

    if not bics:
        # Fallback to k=1 if no valid models found
        gmm = GaussianMixture(n_components=1, random_state=0, n_init=8)
        gmm.fit(X)
        return gmm, 1, np.array([gmm.bic(X)])

    bics = np.array(bics)
    best_idx = int(np.argmin(bics))
    best_k = best_idx + 1
    best_gmm = models[best_idx]
    return best_gmm, best_k, bics


# ======================================================================
# Main analysis for a single file
# ======================================================================

def analyze_top_file(
    top_file: str,
    show_debug_plots: bool = True,
    show_summary_plots: bool = True,
    averaging_radius: int = 1,
    verbose: bool = True,
):
    """
    Full pipeline for a single STM .top file.

    Returns
    -------
    results : dict
        Contains step-height estimates by different methods and their errors.
    """

    # ------------------------------------------------------------------
    # Load & basic preprocessing
    # ------------------------------------------------------------------
    image, meta, header = load_wsxm_top(top_file)

    image = image.astype(float)
    image[image == 0] = np.nan

    x_size_nm = meta["x_amp"] * 1e9   # m → nm
    y_size_nm = meta["y_amp"] * 1e9
    z_scale_nm = 1e9
    image *= z_scale_nm               # height map in nm
    image -= np.nanmin(image)         # shift so min = 0

    rows, cols = image.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    Z = image

    vmin = np.nanmin(image)
    vmax = np.nanmax(image)

    # ------------------------------------------------------------------
    # Segment into 2 regions with GMM on Z
    # ------------------------------------------------------------------
    valid_mask = ~np.isnan(Z)
    Zv = Z[valid_mask].reshape(-1, 1)
    if Zv.size < 10:
        raise RuntimeError(f"[{top_file}] Not enough valid pixels to segment image")

    gmm2 = GaussianMixture(n_components=2, random_state=0, n_init=8).fit(Zv)
    labels = gmm2.predict(Zv)

    # --- parameters of the 2-Gaussian mixture (on height Z) ---
    means_z   = gmm2.means_.ravel()
    vars_z    = gmm2.covariances_.reshape(-1)
    sigmas_z  = np.sqrt(vars_z)
    weights_z = gmm2.weights_.ravel()

    # sort components by mean: comp 1 = lower plateau, comp 2 = higher plateau
    order = np.argsort(means_z)
    means_z   = means_z[order]
    sigmas_z  = sigmas_z[order]
    weights_z = weights_z[order]
    low_label, high_label = order[0], order[1]

    # map labels back into image
    label_img = -np.ones_like(Z, dtype=int)
    label_img[valid_mask] = labels

    mask_low  = label_img == low_label
    mask_high = label_img == high_label


    # ------------------------------------------------------------------
    # Plateau-based method (reference plane from low plateau only)
    # ------------------------------------------------------------------
    # Fit plane only to the low terrace, level everything with that.
    a_ref, b_ref, c_ref = fit_plane(Z, X, Y, mask_low)
    plane_ref = a_ref * X + b_ref * Y + c_ref
    Z_flat = Z - plane_ref

    z_low  = Z_flat[mask_low]
    z_high = Z_flat[mask_high]
    z_low  = z_low[np.isfinite(z_low)]
    z_high = z_high[np.isfinite(z_high)]

    def robust_core(vals):
        if vals.size == 0:
            return vals
        med = np.median(vals)
        mad = 1.4826 * np.median(np.abs(vals - med))
        if not np.isfinite(mad) or mad == 0:
            return vals
        return vals[np.abs(vals - med) < 3 * mad]

    z_low_core  = robust_core(z_low)
    z_high_core = robust_core(z_high)
    if z_low_core.size == 0:
        z_low_core = z_low
    if z_high_core.size == 0:
        z_high_core = z_high

    if z_low_core.size > 0 and z_high_core.size > 0:
        mean_low  = z_low_core.mean()
        mean_high = z_high_core.mean()
        var_low   = z_low_core.var(ddof=1) if z_low_core.size > 1 else 0.0
        var_high  = z_high_core.var(ddof=1) if z_high_core.size > 1 else 0.0
        step_plateau_mean = mean_high - mean_low
        step_plateau_mean_err = np.sqrt(
            (var_low / max(1, z_low_core.size)) +
            (var_high / max(1, z_high_core.size))
        )
        step_plateau_median = np.median(z_high_core) - np.median(z_low_core)
    else:
        step_plateau_mean = np.nan
        step_plateau_mean_err = np.nan
        step_plateau_median = np.nan

    # ------------------------------------------------------------------
    # Segmentation overlay for visualization
    # ------------------------------------------------------------------
    if show_debug_plots:
        seg_overlay = np.zeros((*Z.shape, 4), dtype=float)
        seg_overlay[mask_low]  = [0.0, 0.0, 1.0, 0.25]   # blue
        seg_overlay[mask_high] = [1.0, 0.0, 0.0, 0.25]   # red

        fig = plt.figure(figsize=(6, 5))
        plt.imshow(seg_overlay, origin='lower', extent=[0, x_size_nm, 0, y_size_nm])
        plt.title(f"{top_file}: segmentation (low/high regions)")
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        plt.tight_layout()

    # ------------------------------------------------------------------
    # Raw height map & histogram (no fit overlay here to avoid clutter)
    # ------------------------------------------------------------------
    if show_debug_plots:
        fig = plt.figure(figsize=(6, 5))
        im = plt.imshow(
            image, origin="lower", cmap="viridis",
            extent=[0, x_size_nm, 0, y_size_nm],
            vmin=vmin, vmax=vmax
        )
        plt.colorbar(im, label="Height (nm)")
        plt.title(f"{top_file}: STM height map")
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        plt.tight_layout()

        heights_flat = Z[valid_mask]
        fig = plt.figure(figsize=(6, 5))
        plt.hist(heights_flat, bins=100, density=True,
                 color='blue', alpha=0.4, label="Data")
        plt.xlabel("Height (nm)")
        plt.ylabel("Probability density")
        plt.title(f"{top_file}: height histogram")
        plt.legend(fontsize=8)
        plt.tight_layout()

    # ------------------------------------------------------------------
    # Logistic fits per column
    # ------------------------------------------------------------------
    fit = fit_line_profiles(
        image,
        window_half_width=10,
        min_step_height_nm=0.2,
        rms_rel_tol=0.5,
        rms_abs_tol=0.3,
        max_k=3.0,
        averaging_radius=averaging_radius,
    )

    fit_params = [p[0] if p is not None else None for p in fit]
    fit_errors = [p[1] if p is not None else None for p in fit]

    # Build fitted image
    fitted_image = np.full((rows, cols), np.nan)
    for col, params in enumerate(fit_params):
        if params is None:
            continue
        try:
            fitted_image[:, col] = logistic(np.arange(rows), *params)
        except Exception:
            fitted_image[:, col] = np.nan

    residual = image - fitted_image
    max_res = np.nanmax(np.abs(residual))

    if show_debug_plots:
        # fitted height map
        fig = plt.figure(figsize=(6, 5))
        im = plt.imshow(
            fitted_image, origin="lower", cmap="viridis",
            extent=[0, x_size_nm, 0, y_size_nm],
            vmin=vmin, vmax=vmax
        )
        plt.colorbar(im, label="Fitted height (nm)")
        plt.title(f"{top_file}: fitted logistic per column")
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        plt.tight_layout()

        # residuals
        fig = plt.figure(figsize=(6, 5))
        im = plt.imshow(
            residual, origin="lower", cmap="seismic",
            extent=[0, x_size_nm, 0, y_size_nm],
            vmin=-max_res, vmax=max_res
        )
        plt.colorbar(im, label="Height difference (nm)")
        plt.title(f"{top_file}: residual (data - fit)")
        plt.xlabel("x (nm)")
        plt.ylabel("y (nm)")
        plt.tight_layout()

        # example line profiles
        fig = plt.figure(figsize=(8, 8))
        valid_cols = []
        while len(valid_cols) < min(4, cols):
            random_cols = np.random.choice(cols, size=min(4, cols), replace=False)
            for col in random_cols:
                if len(valid_cols) >= min(4, cols):
                    break
                if np.any(~np.isnan(image[:, col])):
                    valid_cols.append(col)

        for idx, example_col in enumerate(valid_cols[:4], 1):
            plt.subplot(2, 2, idx)
            valid_mask_col = ~np.isnan(image[:, example_col])
            y_vals_nm = np.arange(rows)[valid_mask_col] * (y_size_nm / rows)
            z_vals = image[:, example_col][valid_mask_col]
            plt.plot(y_vals_nm, z_vals, 'k.', label='data')
            if fit_params[example_col] is not None:
                fit_vals = fitted_image[:, example_col]
                plt.plot(
                    np.arange(rows) * (y_size_nm / rows),
                    fit_vals, 'r-', label='fit'
                )
            plt.xlabel('y (nm)')
            plt.ylabel('height z (nm)')
            plt.title(f'Column {example_col} profile')
            plt.legend()

        plt.tight_layout()

    # ------------------------------------------------------------------
    # Column step heights from logistic fits
    # ------------------------------------------------------------------
    heights = np.full(cols, np.nan)
    heights_err = np.full(cols, np.nan)

    for col, (params, errs) in enumerate(zip(fit_params, fit_errors)):
        if params is None:
            continue
        heights[col] = abs(params[1])
        if errs is not None:
            heights_err[col] = errs[1]

    valid_height_mask = np.isfinite(heights)
    heights_all = heights[valid_height_mask]

    if heights_all.size == 0:
        raise RuntimeError(f"[{top_file}] No valid step heights were fitted")

    # robust pre-cleaning via MAD
    med_h = np.median(heights_all)
    mad_h = np.median(np.abs(heights_all - med_h))
    mad_sigma_h = 1.4826 * mad_h
    if np.isfinite(mad_sigma_h) and mad_sigma_h > 0:
        mask_h_core = np.abs(heights_all - med_h) < 3 * mad_sigma_h
    else:
        mask_h_core = np.ones_like(heights_all, dtype=bool)

    h_good = heights_all[mask_h_core]
    err_good = heights_err[valid_height_mask][mask_h_core]
    n_good = h_good.size

    # Column-logistic mean + error (scatter-based)
    if n_good > 0:
        step_col_mean = h_good.mean()
    else:
        step_col_mean = np.nan

    if n_good > 1:
        std_col = h_good.std(ddof=1)
        step_col_mean_err = std_col / np.sqrt(n_good)
    else:
        std_col = np.nan
        step_col_mean_err = np.nan

    # Weighted mean using column fit uncertainties
    good_err_mask = np.isfinite(err_good) & (err_good > 0)
    if np.any(good_err_mask):
        w = 1.0 / (err_good[good_err_mask] ** 2)
        sumw = np.sum(w)
        step_col_wmean = np.sum(w * h_good[good_err_mask]) / sumw
        step_col_wmean_err = np.sqrt(1.0 / sumw)
    else:
        step_col_wmean = np.nan
        step_col_wmean_err = np.nan

    # ------------------------------------------------------------------
    # Histogram of column step heights + GMM (per-file internal structure)
    # ------------------------------------------------------------------
    if show_debug_plots:
        fig = plt.figure(figsize=(6, 4))
        plt.hist(heights_all, bins=90, color='blue', alpha=0.4, density=True, label='All heights')
        plt.hist(h_good, bins=40, color='red', alpha=0.7, density=True, label='After 3·MAD cut')

        mu_fit, sigma_fit = norm.fit(h_good)
        x = np.linspace(h_good.min(), h_good.max(), 200)
        plt.plot(x, norm.pdf(x, mu_fit, sigma_fit), 'k-', linewidth=2,
                 label=f'Gaussian fit: μ={mu_fit:.3f}, σ={sigma_fit:.3f}')

        plt.xlabel("Fitted step height (nm)")
        plt.ylabel("Probability density")
        plt.title(f"{top_file}: histogram of fitted step heights")
        plt.legend(fontsize=8)
        plt.tight_layout()

    # GMM on h_good (to infer layer spacing)
    gmm, best_k, bics = fit_best_gmm(h_good, max_components=7)
    means = gmm.means_.ravel()
    vars_ = gmm.covariances_.ravel()
    sigmas = np.sqrt(vars_)
    weights = gmm.weights_.ravel()

    order = np.argsort(means)
    means = means[order]
    sigmas = sigmas[order]
    weights = weights[order]

    weight_threshold = 0.05
    sig_idx = weights > weight_threshold
    means_sig = means[sig_idx]

    d_est = np.nan
    if means_sig.size >= 2:
        diffs = np.diff(means_sig)
        d_est = np.median(diffs)

    if show_summary_plots:
        fig = plt.figure(figsize=(6, 4))
        plt.hist(h_good, bins=40, density=True, alpha=0.4, label="Data (cleaned)")

        x = np.linspace(h_good.min(), h_good.max(), 500)
        logprob = gmm.score_samples(x.reshape(-1, 1))
        pdf_mix = np.exp(logprob)
        plt.plot(x, pdf_mix, linewidth=2, label=f"GMM mixture (K={best_k})")

        for i, (m, s, w) in enumerate(zip(means, sigmas, weights), start=1):
            comp_pdf = w * norm.pdf(x, m, s)
            plt.plot(x, comp_pdf, linestyle='--', linewidth=1.5,
                     label=f"Comp {i}: μ={m:.3f}, σ={s:.3f}")

        plt.xlabel("Fitted step height (nm)")
        plt.ylabel("Probability density")
        plt.title(f"{top_file}: step heights + Gaussian mixture")
        plt.legend(fontsize=8)
        plt.tight_layout()

    # ------------------------------------------------------------------
    # Plot step heights vs column
    # ------------------------------------------------------------------
    if show_summary_plots or show_debug_plots:
        fig = plt.figure(figsize=(6, 4))
        x_positions = x_size_nm / cols * np.arange(cols)

        plt.errorbar(
            x_positions[valid_height_mask],
            heights_all,
            yerr=heights_err[valid_height_mask],
            fmt='o', color='blue', alpha=0.7,
            elinewidth=1, capsize=2,
            label="column steps"
        )

        if np.isfinite(step_col_mean):
            plt.axhline(
                step_col_mean,
                color='red', linestyle='--',
                label=f'mean = {step_col_mean:.4f} nm'
            )
        if np.isfinite(step_col_wmean):
            plt.axhline(
                step_col_wmean,
                color='green', linestyle=':',
                label=f'w-mean = {step_col_wmean:.4f} nm'
            )

        plt.legend()
        plt.xlabel("x (nm)")
        plt.ylabel("Fitted step height (nm)")
        plt.title(f"{top_file}: fitted step heights vs x")
        plt.tight_layout()

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------
    if verbose:
        print()
        print(f"=== {top_file} ===")
        print(f"Image size: {rows}×{cols} px, {x_size_nm:.1f} × {y_size_nm:.1f} nm")

        print("Plateau-based (global plane):")
        print(f"  N_low = {z_low_core.size}, N_high = {z_high_core.size}")
        print(f"  step_mean = {step_plateau_mean:.4f} ± {step_plateau_mean_err:.4f} nm")
        if np.isfinite(step_plateau_mean):
            print(f"              ≈ {step_plateau_mean/0.335:.3f} layers")
        print(f"  step_median ≈ {step_plateau_median:.4f} nm")

        print("Column-logistic (after 3·MAD cut):")
        print(f"  N_good columns = {n_good}")
        print(f"  step_col_mean       = {step_col_mean:.4f} ± {step_col_mean_err:.4f} nm")
        print(f"  step_col_weighted   = {step_col_wmean:.4f} ± {step_col_wmean_err:.4f} nm")

        print(f"GMM on column steps: best K = {best_k}")
        for i, (m, s, w) in enumerate(zip(means, sigmas, weights), start=1):
            print(f"  Comp #{i}: μ = {m:.4f} nm, σ = {s:.4f} nm, weight = {w:.3f}")
        if not np.isnan(d_est):
            print(f"  inferred layer spacing d ≈ {d_est:.4f} nm "
                  f"(d / 0.335 ≈ {d_est / 0.335:.3f})")

    if show_debug_plots or show_summary_plots:
        plt.show()

    # ------------------------------------------------------------------
    # Collect results to return
    # ------------------------------------------------------------------
    results = {
        "file": top_file,
        "step_plateau_global_mean_nm": step_plateau_mean,
        "step_plateau_global_mean_err_nm": step_plateau_mean_err,
        "step_plateau_global_median_nm": step_plateau_median,
        "step_column_mean_nm": step_col_mean,
        "step_column_mean_err_nm": step_col_mean_err,
        "step_column_weighted_mean_nm": step_col_wmean,
        "step_column_weighted_mean_err_nm": step_col_wmean_err,
        "n_plateau_low": z_low_core.size,
        "n_plateau_high": z_high_core.size,
        "n_good_columns": n_good,
        "d_layer_spacing_nm": d_est,
        "all_heights": h_good,  # All individual heights from this file
        "all_heights_err": err_good,  # Corresponding errors
    }
    return results


# ======================================================================
# Batch processing and combined plots
# ======================================================================

if __name__ == "__main__":

    steps_dir   = "FP/STM/steps"
    plots_dir   = "FP/STM/plots"
    cache_file  = os.path.join(steps_dir, "batch_results_cache.csv")
    top_files   = sorted(glob.glob(os.path.join(steps_dir, "*.top")))

    # ------------------------------------------------------------------
    # Simple switch: cache is used ONLY if you explicitly set this True.
    # ------------------------------------------------------------------
    USE_CACHE = False 

    use_cache = False
    if USE_CACHE and os.path.exists(cache_file):
        cache_mtime = os.path.getmtime(cache_file)
        top_mtimes  = [os.path.getmtime(f) for f in top_files]
        if top_mtimes and max(top_mtimes) < cache_mtime:
            use_cache = True
            print(f"Loading cached results from {cache_file}")

    if use_cache:
        # --------------------------------------------------------------
        # Load scalar results from CSV cache
        # (per-column heights are not cached, so "all_heights" will be NaN)
        # --------------------------------------------------------------
        all_results = []
        with open(cache_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                result = {"file": row["file"]}
                for key in row:
                    if key == "file":
                        continue
                    try:
                        result[key] = float(row[key])
                    except ValueError:
                        result[key] = np.nan
                all_results.append(result)
        print(f"Loaded {len(all_results)} cached results")
    else:
        # --------------------------------------------------------------
        # Fresh analysis of all .top files
        # --------------------------------------------------------------
        print("Running analysis on all files...")
        all_results = []
        for f in top_files:
            try:
                res = analyze_top_file(
                    f,
                    show_debug_plots=True,
                    show_summary_plots=True,
                    averaging_radius=1,
                    verbose=True,
                )
                all_results.append(res)
            except Exception as e:
                print(f"Error processing {f}: {e}")

        # Save scalar results to CSV cache (arrays are omitted automatically)
        if all_results:
            print(f"Saving results to {cache_file}")
            # keep only scalar keys for the CSV
            scalar_keys = [
                k for k in all_results[0].keys()
                if not isinstance(all_results[0][k], np.ndarray)
            ]
            with open(cache_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=scalar_keys)
                writer.writeheader()
                for res in all_results:
                    row = {k: res[k] for k in scalar_keys}
                    writer.writerow(row)

    # ----- Combined analysis over many images -----
    if all_results:
        files = [os.path.basename(r["file"]) for r in all_results]
        idx   = np.arange(len(files))

        def arr(key):
            return np.array([r.get(key, np.nan) for r in all_results], dtype=float)

        # Plateau-based method (kept for completeness but not used in plots)
        # pg_mean   = arr("step_plateau_global_mean_nm")
        # pg_mean_e = arr("step_plateau_global_mean_err_nm")

        col_mean    = arr("step_column_mean_nm")
        col_mean_e  = arr("step_column_mean_err_nm")
        col_wmean   = arr("step_column_weighted_mean_nm")
        col_wmean_e = arr("step_column_weighted_mean_err_nm")

        # ------------------------------------------------------------------
        # Collect ALL individual column heights (extra distribution plot)
        # ------------------------------------------------------------------
        all_heights_combined     = []
        all_heights_err_combined = []
        heights_per_image        = []   # list of arrays, one per file

        for r in all_results:
            h_arr = r.get("all_heights", None)
            e_arr = r.get("all_heights_err", None)
            if isinstance(h_arr, np.ndarray) and h_arr.size:
                heights_per_image.append(h_arr)
                all_heights_combined.extend(h_arr.flatten())
                if isinstance(e_arr, np.ndarray) and e_arr.size:
                    all_heights_err_combined.extend(e_arr.flatten())

        all_heights_combined     = np.array(all_heights_combined)
        all_heights_err_combined = np.array(all_heights_err_combined)

        h0_lit = 0.335  # literature single layer height (nm)

        # Helper: numeric summary you can quote in the text
        def global_summary(name, heights, heights_e=None):
            mask = np.isfinite(heights)
            h = heights[mask]
            if h.size == 0:
                return

            mean = h.mean()
            std  = h.std(ddof=1) if h.size > 1 else np.nan
            stderr = std / np.sqrt(h.size) if h.size > 1 else np.nan
            print(f"{name}: unweighted mean = {mean:.4f} ± {stderr:.4f} nm  (N = {h.size})")

            if heights_e is not None:
                mask_w = mask & np.isfinite(heights_e) & (heights_e > 0)
                if np.any(mask_w):
                    h_w = heights[mask_w]
                    e_w = heights_e[mask_w]
                    w   = 1.0 / (e_w ** 2)
                    h_bar = np.sum(w * h_w) / np.sum(w)
                    h_err = np.sqrt(1.0 / np.sum(w))
                    print(f"    weighted by per-image σ: {h_bar:.4f} ± {h_err:.4f} nm")

        methods = [
            ("Column-logistic mean",          col_mean,  col_mean_e),
            ("Column-logistic weighted mean", col_wmean, col_wmean_e),
        ]

        # ------------------------------------------------------------------
        # Histograms of per-image means vs. all individual heights
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Left: per-image means
        ax = axes[0]
        h_plot = col_mean[np.isfinite(col_mean)]
        ax.hist(h_plot, bins=70, alpha=0.7, color='blue', label='Per-image means')
        ax.set_xlabel("Step height (nm)")
        ax.set_ylabel("Count")
        ax.set_title("Per-image column-logistic means")

        if h_plot.size:
            h_max = float(h_plot.max())
            n_max = int(np.ceil(h_max / h0_lit))
            for n in range(1, n_max + 1):
                ax.axvline(n * h0_lit, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.legend()
        plt.savefig(os.path.join(plots_dir, "per_image_means_histogram.pdf"))

        # Right: all individual heights (combined distribution)
        ax = axes[1]
        if all_heights_combined.size > 0:
            ax.hist(all_heights_combined, bins=100, alpha=0.7, color='red',
                    label='All column heights')
            ax.set_xlabel("Step height (nm)")
            ax.set_ylabel("Count")
            ax.set_title(f"All fitted column heights (N={all_heights_combined.size})")
            h_max = float(np.nanmax(all_heights_combined))
            n_max = int(np.ceil(h_max / h0_lit))
            for n in range(1, n_max + 1):
                ax.axvline(n * h0_lit, linestyle=":", linewidth=0.8, alpha=0.5)
            ax.legend()
        plt.savefig(os.path.join(plots_dir, "all_heights_combined_histogram.pdf"))
        fig.tight_layout()

        # ------------------------------------------------------------------
        # EXTRA: per-image distributions of all column heights (boxplot)
        # ------------------------------------------------------------------
        if heights_per_image:
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111)

            # boxplot expects a list of 1D arrays
            bp = ax.boxplot(heights_per_image,
                            positions=np.arange(len(heights_per_image)),
                            showfliers=True)

            ax.set_xticks(np.arange(len(files)))
            ax.set_xticklabels(files, rotation=45, ha="right")
            ax.set_ylabel("Step height (nm)")
            ax.set_title("Distribution of all fitted step heights per image")

            # guideline lines at n·h0
            try:
                h_max = float(np.nanmax(all_heights_combined))
                if np.isfinite(h_max) and h_max > 0:
                    n_max = int(np.ceil(h_max / h0_lit))
                    for n in range(1, n_max + 1):
                        ax.axhline(n * h0_lit, linestyle=":", linewidth=0.8, color="gray")
            except Exception:
                pass

            fig.tight_layout()
            plt.savefig(os.path.join(plots_dir, "all_heights_per_image_boxplot.pdf"))
        # ------------------------------------------------------------------
        # Step height vs. image (means + weighted means)
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(9, 5))

        if np.any(np.isfinite(col_mean)):
            plt.errorbar(idx, col_mean, yerr=col_mean_e,
                         fmt="s-", label="Column-logistic mean")

        if np.any(np.isfinite(col_wmean)):
            plt.errorbar(idx, col_wmean, yerr=col_wmean_e,
                         fmt="^-", label="Column-logistic weighted mean")

        plt.xticks(idx, files, rotation=45, ha="right")
        plt.xlabel("Image index / file")
        plt.ylabel("Step height (nm)")
        plt.title("Step heights per image (column-logistic methods)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "step_heights_per_image.pdf"))

        # ------------------------------------------------------------------
        # Same heights, sorted: spread + outliers
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(6, 5))
        order = np.argsort(col_mean)
        plt.errorbar(
            np.arange(len(order)), col_mean[order],
            yerr=col_mean_e[order],
            fmt="o", alpha=0.7, linestyle="None",
        )

        try:
            h_max = float(np.nanmax(col_mean[order]))
            if np.isfinite(h_max) and h_max > 0:
                n_max = int(np.ceil(h_max / h0_lit))
                for n in range(1, n_max + 1):
                    plt.axhline(
                        n * h0_lit,
                        linestyle=":", linewidth=0.8, color="gray",
                        label=(f"{h0_lit:.3f} nm multiples" if n == 1 else "_nolegend_")
                    )
        except Exception:
            pass

        plt.xlabel("Index (sorted by step height)")
        plt.ylabel("Step height (nm)")
        plt.title("Column-logistic mean (sorted)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "column_logistic_mean_sorted.pdf"))
        # ------------------------------------------------------------------
        # Consistency with integer multiples of h0 for the two methods
        # ------------------------------------------------------------------
        for (method_name, heights, heights_e) in methods:
            fig = plt.figure(figsize=(10, 4))
            fig.suptitle(method_name)

            h = heights[np.isfinite(heights)]
            e = heights_e[np.isfinite(heights_e)]
            if h.size == 0:
                continue

            ratio  = h / h0_lit
            n_near = np.rint(ratio).astype(int)
            delta  = ratio - n_near  # dimensionless deviation

            ax1 = plt.subplot(1, 2, 1)
            ax1.hist(delta, bins=20)
            ax1.set_xlabel(r"$h/h_0 - \mathrm{round}(h/h_0)$")
            ax1.set_ylabel("Count")
            ax1.set_title("Distance to nearest integer multiple")
            ax1.axvline(0.0, color="k", linewidth=0.8)

            ax2 = plt.subplot(1, 2, 2)
            ax2.errorbar(n_near, h, yerr=e, fmt="o", linestyle="None")
            x_line = np.array([0, n_near.max() + 1])
            ax2.plot(x_line, x_line * h0_lit, "--")
            ax2.set_xlabel("Nearest integer $n$")
            ax2.set_ylabel("Step height $h$ (nm)")
            ax2.set_title(r"$h$ vs. $n \cdot h_0$")

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            plt.savefig(os.path.join(
                plots_dir,
                f"consistency_with_h0_{method_name.replace(' ', '_').lower()}.pdf"
            ))
            

        # ------------------------------------------------------------------
        # Global numbers you can quote in the report
        # ------------------------------------------------------------------
        for name, h, h_e in methods:
            global_summary(name, h, h_e)

        if all_heights_combined.size > 0:
            # optional global summary for all individual heights
            mean_all = np.nanmean(all_heights_combined)
            std_all  = np.nanstd(all_heights_combined, ddof=1)
            stderr_all = std_all / np.sqrt(all_heights_combined.size)
            print(
                f"All column heights combined: "
                f"{mean_all:.4f} ± {stderr_all:.4f} nm (N = {all_heights_combined.size})"
            )
            
        plt.savefig(os.path.join(plots_dir, "combined_step_heights_summary.pdf"))
        #plt.show()
