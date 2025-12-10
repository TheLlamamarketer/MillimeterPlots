import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import glob
import os




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


def logistic(x, z0, h, x0, k):
    return z0 + h * expit(k * (x - x0))

def fit_line_profiles(image, min_points = 10, min_step_height_nm = None, grad_sigma = 2, window_half_width = 30, max_k = 5.0, rms_rel_tol = 0.3, rms_abs_tol = 0.2, averaging_radius:int = 0):
    
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


def analyze_top_file(
    top_file: str,
    show_debug_plots: bool = True,
    show_summary_plots: bool = True,
    averaging_radius: int = 1,
    verbose: bool = True,
):
    """
    Full pipeline for a single STM .top file:
    - Load and convert to nm.
    - Segment into two height regions with GMM and plane-fit each.
    - Fit logistic steps column-wise.
    - Compute step height statistics and GMM over step heights.
    - Optionally make diagnostic and summary plots.

    Returns
    -------
    results : dict
        Contains step-height statistics, GMM parameters, etc.
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
    # Segment into 2 regions with GMM on Z and plane-fit each
    # ------------------------------------------------------------------
    valid_mask = ~np.isnan(Z)
    Zv = Z[valid_mask].reshape(-1, 1)
    if Zv.size < 10:
        raise RuntimeError(f"[{top_file}] Not enough valid pixels to segment image")

    gmm2 = GaussianMixture(n_components=2, random_state=0, n_init=8).fit(Zv)
    labels = gmm2.predict(Zv)

    # --- parameters of the 2-Gaussian mixture (on height Z) ---
    means_z   = gmm2.means_.ravel()                 # shape (2,)
    vars_z    = gmm2.covariances_.reshape(-1)       # shape (2,) because it's 1D
    sigmas_z  = np.sqrt(vars_z)
    weights_z = gmm2.weights_.ravel()               # shape (2,)

    # sort components by mean: comp 1 = lower plateau, comp 2 = higher plateau
    order = np.argsort(means_z)
    means_z   = means_z[order]
    sigmas_z  = sigmas_z[order]
    weights_z = weights_z[order]

    low_label,  high_label  = order[0], order[1]

    # map labels back into image
    label_img = -np.ones_like(Z, dtype=int)
    label_img[valid_mask] = labels

    mask_low  = label_img == low_label
    mask_high = label_img == high_label

    # --- plane fit on each region ---
    a_low,  b_low,  c_low  = fit_plane(Z, X, Y, mask_low)
    a_high, b_high, c_high = fit_plane(Z, X, Y, mask_high)

    plane_low  = a_low  * X + b_low  * Y + c_low
    plane_high = a_high * X + b_high * Y + c_high

    image_corr = np.full_like(image, np.nan)
    image_corr[mask_low]  = image[mask_low]  - plane_low[mask_low]
    image_corr[mask_high] = image[mask_high] - plane_high[mask_high]

    # plateau heights after plane removal
    h_low  = np.nanmedian(image_corr[mask_low])
    h_high = np.nanmedian(image_corr[mask_high])
    step_height_plane = h_high - h_low  

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
    # Raw height map & histogram + 2-Gaussian GMM overlay
    # ------------------------------------------------------------------
    if show_debug_plots:
        # height map
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
        plt.title(f"{top_file}: height histogram + 2-Gaussian GMM")
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
                fit_vals = fitted_image[:, example_col] * 1.0
                plt.plot(
                    np.arange(rows) * (y_size_nm / rows),
                    fit_vals, 'r-', label='fit'
                )
            plt.xlabel('y (nm)')
            plt.ylabel('height z (nm)')
            plt.title(f'Column {example_col} profile')
            plt.legend()

        plt.tight_layout()


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
    
    


    # ------------------------------------------------------------------
    # Histogram of step heights + GMM
    # ------------------------------------------------------------------
    if heights_all.size == 0:
        raise RuntimeError(f"[{top_file}] No valid step heights were fitted")

    
    # robust pre-cleaning via MAD
    med = np.median(heights_all)
    mad = np.median(np.abs(heights_all - med))
    mad_sigma = 1.4826 * mad

    mask_mad = np.abs(heights_all - med) < 3 * mad_sigma
    h_good = heights_all[mask_mad]

    mean_simple = h_good.mean()
    err_good = heights_err[valid_height_mask][mask_mad]
    good_for_weight = np.isfinite(err_good) & (err_good > 0)
    if np.any(good_for_weight):
        w = 1.0 / err_good[good_for_weight] ** 2
        mean_simple_weighted = np.average(h_good[good_for_weight], weights=w)
    else:
        mean_simple_weighted = np.nan
    std_simple = h_good.std(ddof=1)
    stderr_simple = std_simple / np.sqrt(h_good.size)
    
    
    #  simple Gaussian fit plot
    if show_debug_plots:
        fig = plt.figure(figsize=(6, 4))
        plt.hist(heights_all, bins=90, color='blue', alpha=0.4, density=True, label='All heights')
        plt.hist(h_good, bins=40, color='red', alpha=0.7, density=True, label='After 3·MAD cut')

        mu, sigma = norm.fit(h_good)
        x = np.linspace(h_good.min(), h_good.max(), 200)
        plt.plot(x, norm.pdf(x, mu, sigma), 'k-', linewidth=2,
                 label=f'Gaussian fit: μ={mu:.3f}, σ={sigma:.3f}')

        plt.xlabel("Fitted step height (nm)")
        plt.ylabel("Probability density")
        plt.title(f"{top_file}: histogram of fitted step heights")
        plt.legend(fontsize=8)
        plt.tight_layout()




    gmm, best_k, bics = fit_best_gmm(h_good, max_components=7)


    means = gmm.means_.ravel()
    vars_ = gmm.covariances_.ravel()
    sigmas = np.sqrt(vars_)
    weights = gmm.weights_.ravel()

    # sort by mean
    order = np.argsort(means)
    means = means[order]
    sigmas = sigmas[order]
    weights = weights[order]
    
    # estimate fundamental spacing d from significant components
    weight_threshold = 0.05
    sig_idx = weights > weight_threshold
    means_sig = means[sig_idx]

    d_est = np.nan
    if means_sig.size >= 2:
        diffs = np.diff(means_sig)
        d_est = np.median(diffs)



     # summary mixture plot (good "heights plot" to keep when running many files)
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
    
    
    # plot step heights vs column
    if show_summary_plots or show_debug_plots:
        fig = plt.figure(figsize=(6, 4))
        x_positions = x_size_nm / cols * np.arange(cols)
        # for plotting errors, just mask NaNs
        err_mask = valid_height_mask & np.isfinite(heights_err)
        yerr = np.where(err_mask, heights_err, np.nan)

        plt.errorbar(
            x_positions[valid_height_mask],
            heights_all,
            yerr=yerr[valid_height_mask],
            fmt='o', color='blue', alpha=0.7,
            elinewidth=1, capsize=2
        )
        
        # weighted mean (only where errors are positive and finite)
        mean_weighted = np.nan
        good_for_weight =  np.isfinite(heights_err) & (heights_err > 0)
        if np.any(good_for_weight):
            w = 1.0 / heights_err[good_for_weight] ** 2
            mean_weighted = np.average(heights_all[good_for_weight], weights=w)
        else:
            mean_weighted = mean_simple
            
        plt.axhline(
            y=mean_weighted,
            color='red', linestyle='--',
            label=f'h_{{all}} = {mean_weighted:.4f} nm'
        )
        plt.axhline(
            y=mean_simple_weighted,
            color='green', linestyle=':',
            label=f'h_{{significant}} = {mean_simple_weighted:.4f} nm'
        )
        
        
        
        plt.legend()
        plt.xlabel("x (nm)")
        plt.ylabel("Fitted step height (nm)")
        plt.title(f"{top_file}: fitted step heights vs x")
        plt.tight_layout()
    
    
    if verbose:
        print()
        print(f"=== {top_file} ===")
        print(f"Image size: {rows}*{cols} px, {x_size_nm:.1f} * {y_size_nm:.1f} nm")
        print(f"Plane-fit step height (low→high region): {step_height_plane:.4f} nm")
        print(f"Logistic step heights after 3·MAD cut: N = {h_good.size}")
        print(f"   h = {mean_simple:.4f} ± {stderr_simple:.4f} nm (simple mean)")
        if np.isfinite(mean_weighted):
            print(f"   h_{{weighted}} = {mean_weighted:.4f} nm (weighted mean)")
        print(f"GMM best K = {best_k}")
        for i, (m, s, w) in enumerate(zip(means, sigmas, weights), start=1):
            print(f"  Comp #{i}: μ = {m:.4f} nm, σ = {s:.4f} nm, weight = {w:.3f}")
        if not np.isnan(d_est):
            print(f"Estimated layer spacing from GMM: d ≈ {d_est:.4f} nm "
                  f"(d / 0.335 nm ≈ {d_est / 0.335:.3f})")
        

    # show every plot at the same time
    if show_debug_plots or show_summary_plots:
        plt.show()
    else:
        plt.close()


# only run when the file is executed directly, so that when something is imported no code is run
if __name__ == "__main__":
    
    steps_dir = "FP/STM/steps"
    top_files = sorted(glob.glob(os.path.join(steps_dir, "*.top")))

    for f in top_files:
        try:
            analyze_top_file(
                f,
                show_debug_plots=True,
                show_summary_plots=True,
                averaging_radius=1,
                verbose=True,
            )
        except Exception as e:
            print(f"Error processing {f}: {e}")
