import numpy as np
import matplotlib.pyplot as plt
import colour
from functools import lru_cache
from matplotlib.colors import ListedColormap
import matplotlib

# =========================
# Config & precomputation
# =========================

CMFS_1931 = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
ILL_D65   = colour.SDS_ILLUMINANTS['D65']

SHAPE = colour.SpectralShape(380, 780, 1)

def set_base_resolution(step_nm=0.1, start=380.0, end=780.0):
    """Rebuild the precomputed XYZ tables on a new spectral grid.
    Use step_nm=0.5 or 0.1 if you want the CMF/illuminant interpolation
    done by 'colour' (Sprague) instead of our fast linear lookups."""
    global SHAPE, cmfs, ill, XYZ_AE, XYZ_D65
    SHAPE = colour.SpectralShape(start, end, step_nm)
    cmfs  = CMFS_1931.copy().align(SHAPE)
    ill   = ILL_D65.copy().align(SHAPE)
    # Ensure both arrays have the same length after alignment
    min_len = min(cmfs.values.shape[0], ill.values.shape[0])
    XYZ_AE  = cmfs.values[:min_len].astype(float)                      # equal energy
    XYZ_D65 = (XYZ_AE * ill.values[:min_len, None]).astype(float)      # D65 weighted

    if '_exposure' in globals():
        _exposure.cache_clear()

# initialize
step_size = 0.01
set_base_resolution(step_size)

# XYZ → linear RGB matrices
M_XYZ2LIN_SRGB = np.array([[ 3.2404542,-1.5371385,-0.4985314],
                           [-0.9692660, 1.8760108, 0.0415560],
                           [ 0.0556434,-0.2040259, 1.0572252]], dtype=float)

M_XYZ2LIN_P3   = np.array([[ 2.4934969,-0.9313836,-0.4027108],
                           [-0.8294889, 1.7697341, 0.0295424],
                           [ 0.0358458,-0.0761724, 0.9568845]], dtype=float)

# =========================
# Transfer functions
# =========================

def srgb_eotf(lin):
    lin = np.asarray(lin, dtype=float)
    out = np.empty_like(lin)
    m = lin <= 0.0031308
    out[m]  = 12.92 * lin[m]
    out[~m] = 1.055 * np.power(lin[~m], 1/2.4) - 0.055
    return out

# =========================
# Continuous XYZ lookup
# =========================

def _xyz_at_wavelength(wl_nm: float, use_D65: bool = True):
    # Evaluate CMFs and illuminant at an arbitrary wavelength.
    if wl_nm < SHAPE.start or wl_nm > SHAPE.end:
        return np.array([0.0, 0.0, 0.0], float)

    XYZ = np.asarray(CMFS_1931[wl_nm], dtype=float)     # Sprague interp
    if use_D65:
        return XYZ * float(ILL_D65[wl_nm])              # scalar D65 at wl
    return XYZ

# =========================
# Gamut mapping
# =========================

def _project_oklab_to_gamut(Lab, M_rgb, require_upper_bound=False):
    L, a, b = map(float, Lab)
    XYZ = colour.Oklab_to_XYZ([L, a, b])
    lin = (M_rgb @ XYZ).astype(float)
    ok  = (lin >= 0).all() if not require_upper_bound else ((lin >= 0).all() and (lin <= 1).all())
    if ok:
        return np.clip(lin, 0.0, 1.0) if require_upper_bound else np.maximum(lin, 0.0)

    XYZ_n = colour.Oklab_to_XYZ([L, 0.0, 0.0])
    lin_n = (M_rgb @ XYZ_n).astype(float)

    lo, hi = 0.0, 1.0
    best = None
    for _ in range(24):
        s = (lo + hi) * 0.5
        XYZ_s = colour.Oklab_to_XYZ([L, s*a, s*b])
        lin_s = (M_rgb @ XYZ_s).astype(float)
        ok = (lin_s >= 0).all() if not require_upper_bound else ((lin_s >= 0).all() and (lin_s <= 1).all())
        if ok:
            lo, best = s, lin_s
        else:
            hi = s
    if best is None:
        best = np.maximum(lin_n, 0.0)
    return np.clip(best, 0.0, 1.0) if require_upper_bound else np.maximum(best, 0.0)

def _map_lin_from_XYZ(XYZ, M_rgb, gamut_map):
    lin0 = (M_rgb @ XYZ).astype(float)
    if gamut_map == "clip":
        return np.maximum(lin0, 0.0)
    if gamut_map == "oklab":
        L, a, b = colour.XYZ_to_Oklab(XYZ)
        return _project_oklab_to_gamut([L, a, b], M_rgb, require_upper_bound=False)
    return np.clip(lin0, 0.0, 1.0)

# =========================
# Oklab-only post processing
# =========================

def _slice_from_wls(wls, wl0, wl1):
    i0 = int(np.searchsorted(wls, wl0, side="left"))
    i1 = int(np.searchsorted(wls, wl1, side="right") - 1)
    if i1 <= i0 or i0 >= len(wls) or i1 < 0:
        return None, None
    i0 = max(0, min(i0, len(wls)-1))
    i1 = max(0, min(i1, len(wls)-1))
    return i0, i1

def fix_blue_dip(img, wls, wl0=459, wl1=525, strength=0.95, ease=True):
    i0, i1 = _slice_from_wls(wls, wl0, wl1)
    if i0 is None: return img
    seg = img[0, i0:i1+1, 2].copy()
    t = np.linspace(0.0, 1.0, seg.size)
    if ease: t = t*t*(3 - 2*t)  # smoothstep
    base   = (1.0 - t) * seg[0] + t * seg[-1]
    raised = seg + strength * np.maximum(base - seg, 0.0)  # only lift
    img[:, i0:i1+1, 2] = raised
    return img

def fix_green_bump(img, wls, wl0=459, wl1=510, strength=0.90, ease=True):
    i0, i1 = _slice_from_wls(wls, wl0, wl1)
    if i0 is None: return img
    seg = img[0, i0:i1+1, 1].copy()
    t = np.linspace(0.0, 1.0, seg.size)
    if ease: t = t*t*(3 - 2*t)
    base    = (1.0 - t) * seg[0] + t * seg[-1]
    lowered = seg - strength * np.maximum(seg - base, 0.0)  # only lower
    img[:, i0:i1+1, 1] = lowered
    return img

def _postprocess_if_oklab(img, gamut_map, wls):
    if gamut_map == "oklab":
        img = fix_blue_dip(img, wls)
        img = fix_green_bump(img, wls)
    return img
# =========================
# Exposure (global scale)
# =========================

@lru_cache(maxsize=None)
def _exposure(space: str, use_D65: bool, gamut_map: str, samatplotlibe_step=0.25):
    M = M_XYZ2LIN_P3 if space.upper()=="P3" else M_XYZ2LIN_SRGB
    wls = np.arange(SHAPE.start, SHAPE.end + 1e-9, samatplotlibe_step)
    lin_max = 0.0
    for w in wls:
        XYZ = _xyz_at_wavelength(w, use_D65=use_D65)
        lin = _map_lin_from_XYZ(XYZ, M, gamut_map)
        lin_max = max(lin_max, float(lin.max()))
    return 1.0/lin_max if lin_max > 0 else 1.0

# =========================
# Public API
# =========================

def wavelength_to_rgb(wl_nm: float, *, space="sRGB", brightness="D65",
                      gamut_map="oklab", encode=True):
    """Spectral line at wl_nm → RGB (continuous wl_nm, no 1 nm constraint)."""
    XYZ = _xyz_at_wavelength(wl_nm, use_D65=(brightness=="D65"))
    M   = M_XYZ2LIN_P3 if space.upper()=="P3" else M_XYZ2LIN_SRGB
    lin = _map_lin_from_XYZ(XYZ, M, gamut_map)
    lin *= _exposure(space.upper(), brightness=="D65", gamut_map)
    return srgb_eotf(lin) if encode else np.clip(lin, 0.0, 1.0)

def render_spectrum(*, start=380, end=780, step=0.1, space="sRGB",
                    brightness="D65", gamut_map="oklab", H=40):
    wls = np.arange(start, end + 1e-9, step)
    rgb = np.array([wavelength_to_rgb(w, space=space, brightness=brightness,
                                      gamut_map=gamut_map, encode=True)
                    for w in wls], dtype=float)
    img = np.tile(rgb.reshape(1, -1, 3), (H, 1, 1))
    return _postprocess_if_oklab(img, gamut_map, wls), wls

def lines_to_rgb(wavelengths_nm, intensities=1.0, **kwargs):
    wl = np.atleast_1d(wavelengths_nm).astype(float)
    k  = np.atleast_1d(intensities).astype(float)
    if k.size == 1: k = np.repeat(k, wl.size)
    lin = np.zeros(3, float)
    for w, a in zip(wl, k):
        lin += a * wavelength_to_rgb(w, encode=False, **kwargs)
    return srgb_eotf(np.clip(lin, 0.0, 1.0))


def draw_lines_on_strip(img, grid, wls, amps, sigma_nm=0.05,
                        space="sRGB", brightness="equal", gamut_map="oklab"):
    col_wls = grid
    # Ensure all wavelengths are floats, skip invalid ones
    for wl, amp in zip(wls, amps):
        try:
            wl_float = float(wl)
        except (ValueError, TypeError):
            continue
        if amp is None or amp <= 0.0:
            continue
        w = np.exp(-0.5*((col_wls - wl_float)/sigma_nm)**2)
        col = wavelength_to_rgb(wl_float, encode=True, space=space,
                                brightness=brightness, gamut_map=gamut_map)
        img += (amp * w)[None, :, None] * col[None, None, :]
    return np.clip(img, 0.0, 1.0)


def show_source_strip(title, wls, amps, *, step=0.1, **kwargs):
    base, grid = render_spectrum(step=step, space="sRGB",
                                 brightness="equal",
                                 gamut_map=kwargs.get("gamut_map","oklab"))
    img  = (base * 0.1).copy()
    img = draw_lines_on_strip(img, grid, wls, amps, space="sRGB", brightness="equal", gamut_map=kwargs.get("gamut_map", "oklab"))

    mix = lines_to_rgb(wls, amps, **kwargs)

    fig = plt.figure(figsize=(10, 4), constrained_layout=True)
    gs  = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 6])

    ax0 = fig.add_subplot(gs[0])
    patch = np.ones((20, 120, 3)) * mix
    ax0.imshow(patch, aspect='auto'); ax0.axis("off")

    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(img, extent=[grid[0], grid[-1], 0, 1], aspect='auto')
    ax1.set_xlabel("nm"); ax1.set_yticks([])
    fig.suptitle(title, y=0.98)  # keep title away from patch
    plt.show()



# =========================
# Create a colormap
# =========================

def build_spectrum_cmap(
    name="spectrum_wavelengths",
    *,
    start=380.0, end=780.0, step=0.01,
    space="sRGB", brightness="D65", gamut_map="oklab",
    register=True
):
    """
    Build a Matplotlib ListedColormap sampling your render_spectrum().
    Returns (cmap, norm).

    norm maps wavelength in nm → [0,1], so you can do:
      plt.scatter(..., c=wl_nm, cmap=cmap, norm=norm)
    """
    img, wls = render_spectrum(start=start, end=end, step=step,
                               space=space, brightness=brightness,
                               gamut_map=gamut_map, H=1)
    colors = np.clip(img[0], 0.0, 1.0)               # shape (N, 3)
    cmap = ListedColormap(colors, name=name, N=colors.shape[0])
    norm = matplotlib.colors.Normalize(vmin=wls[0], vmax=wls[-1])

    if register:
        try:
            matplotlib.colormaps.register(cmap, name=name, force=True)
        except Exception:
            # older Matplotlib fallback
            plt.register_cmap(name=name, cmap=cmap)
    return cmap, norm

def wavelength_to_rgba_cm(wl_nm, cmap, norm):
    """Convenience: wavelength (nm) → RGBA via the colormap."""
    return cmap(norm(wl_nm))

def save_cmap_lut(path, *, start=380.0, end=780.0, step=0.1,
                  space="sRGB", brightness="D65", gamut_map="oklab"):
    """
    Persist the LUT to disk so you can reload without recomputing.
    Saves a .npz with 'wls' and 'rgbs' arrays.
    """
    img, wls = render_spectrum(start=start, end=end, step=step,
                               space=space, brightness=brightness,
                               gamut_map=gamut_map, H=1)
    np.savez(path, wls=wls, rgbs=np.clip(img[0], 0.0, 1.0))

def load_cmap(path, name="spectrum_from_file"):
    """Load a LUT saved by save_cmap_lut and return (cmap, norm)."""
    data = np.load(path)
    wls  = data["wls"]
    rgbs = data["rgbs"]
    cmap = ListedColormap(rgbs, name=name, N=rgbs.shape[0])
    norm = matplotlib.colors.Normalize(vmin=wls[0], vmax=wls[-1])
    return cmap, norm


#save_cmap_lut("colormaps/spectrum_wavelengths_0p01.npz", step=0.01)




# =========================
# Example usage
# =========================

# 1) A spectrum strip (D65 prism, sRGB monitor, Oklab gamut mapping)

def show_spectrum_strip():
    img, wls = render_spectrum(space="sRGB", brightness="D65", gamut_map="oklab", step=0.05)  # "clip", "oklab", "oklab_strict", "oklab_soft", "clip_Y"

    fig, axs = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [1, 2]}, sharex=True)

    # Top: RGB components
    axs[0].plot(wls, img[0,:,0], 'r', label='Red')
    axs[0].plot(wls, img[0,:,1], 'g', label='Green')
    axs[0].plot(wls, img[0,:,2], 'b', label='Blue')
    axs[0].set_xlabel("Wavelength (nm)")
    axs[0].set_ylabel("sRGB Value")
    axs[0].set_title("RGB Components of Spectrum")
    axs[0].legend()

    # Bottom: Spectrum strip
    axs[1].imshow(np.tile(img[0:1, :, :], (120, 1, 1)), extent=[380,780,0,1], aspect='auto')
    axs[1].set_title("Spectrum Strip (D65, sRGB, Oklab Gamut)")
    axs[1].set_yticks([])
    axs[1].set_xlabel("nm")

    plt.tight_layout()
    plt.show()

    stepsize = 0.05  # step size for the source strips

    show_source_strip("Sodium lamp (Na D)", [588.995, 589.592, 568.819, 615.422], [1.0, 0.6, 0.3, 0.2],
                    step=stepsize)

    # Mercury (common calibration set)
    show_source_strip("Mercury (Hg)", [365.015, 404.656, 435.833, 491.607, 546.074, 576.959, 579.066],
                    [0.3, 0.5, 0.6, 0.4, 1.0, 0.9, 0.7],
                    step=stepsize)

    # Hydrogen Balmer (lab discharge)
    show_source_strip("Hydrogen Balmer", [397.01, 410.17, 434.05, 486.13, 656.28],
                    [0.15, 0.25, 0.40, 0.80, 1.00],
                    step=stepsize)

    # Neon sign (red-orange) – approximate bright lines
    show_source_strip("Neon (approx.)",
                    [540.056, 585.249, 588.189, 594.483, 603.0, 607.433, 609.616, 614.306, 621.728, 626.649, 633.443, 640.225, 650.653, 659.895, 692.947, 703.241],
                    [0.2,   1.0,    0.8,   0.5,   0.7,   0.3,   0.8,   0.4,   0.3,   0.2,   0.2,   0.9,   0.6,   0.5,   0.4,   0.6],
                    step=stepsize)



    def gaussian(w, mu, sigma):
        return np.exp(-0.5*((w-mu)/sigma)**2)

    w = np.arange(SHAPE.start, SHAPE.end+1)
    pump = 0.7*gaussian(w, 450, 8)          # blue pump
    phos = 1.0*gaussian(w, 560, 60)         # yellow phosphor
    spd  = pump + phos
    # visualize as continuous SPD by summing many lines
    amps = spd / spd.max()
    show_source_strip("Phosphor white LED (model)", w.tolist(), amps.tolist(),
                    space="sRGB", brightness="equal", gamut_map="oklab")

if __name__ == "__main__":
    show_spectrum_strip()
