import numpy as np
import matplotlib.pyplot as plt
import math
import colour

wavelengths = np.linspace(390, 780, 2000)

cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
ill  = colour.SDS_ILLUMINANTS['D65']
M_XYZ2sRGB = colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB

def wl_to_srgb_2015(l_nm):
    XYZ = cmfs[l_nm] * 1.0
    rgb_lin = np.dot(M_XYZ2sRGB, XYZ)
    rgb = colour.cctf_encoding(np.clip(rgb_lin, 0, None))
    return tuple((rgb*255+0.5).astype(int))





def wavelength_to_rgb(wavelength):
    if wavelength < 400 or wavelength > 780 or math.isnan(wavelength):
        return (0, 0, 0)
    r, g, b = 0.0, 0.0, 0.0
    if 400 <= wavelength < 410:
        t = (wavelength - 400.0) / (410.0 - 400.0)
        r = (0.33 * t) - (0.20 * t * t)
    elif 410 <= wavelength < 475:
        t = (wavelength - 410.0) / (475.0 - 410.0)
        r = 0.14 - (0.13 * t * t)
    elif 545 <= wavelength < 595:
        t = (wavelength - 545.0) / (595.0 - 545.0)
        r = (1.98 * t) - (t * t)
    elif 595 <= wavelength < 650:
        t = (wavelength - 595.0) / (650.0 - 595.0)
        r = 0.98 + (0.06 * t) - (0.40 * t * t)
    elif 650 <= wavelength < 680:
        t = (wavelength - 650.0) / (700.0 - 650.0)
        r = 0.65 - (0.84 * t) + (0.20 * t * t)
    elif 680 <= wavelength:
        r = 1.307*10e13*math.e**(-0.05001*wavelength)
    if 415 <= wavelength < 475:
        t = (wavelength - 415.0) / (475.0 - 415.0)
        g = (0.80 * t * t)
    elif 475 <= wavelength < 590:
        t = (wavelength - 475.0) / (590.0 - 475.0)
        g = 0.8 + (0.76 * t) - (0.80 * t * t)
    elif 585 <= wavelength < 639:
        t = (wavelength - 585.0) / (639.0 - 585.0)
        g = 0.84 - (0.84 * t)
    if 400 <= wavelength < 475:
        t = (wavelength - 400.0) / (475.0 - 400.0)
        b = (2.20 * t) - (1.50 * t * t)
    elif 475 <= wavelength < 560:
        t = (wavelength - 475.0) / (560.0 - 475.0)
        b = 0.7 - (t) + (0.30 * t * t)
    R = int(np.clip(r * 255, 0, 255))
    G = int(np.clip(g * 255, 0, 255))
    B = int(np.clip(b * 255, 0, 255))
    return (R, G, B)








def _g(lambda_nm, mu, t1, t2):
    lambda_nm = np.asarray(lambda_nm)
    sigma = np.where(lambda_nm < mu, t1, t2)
    return np.exp(-0.5 * (sigma * (lambda_nm - mu)) ** 2)

def xyz_from_wavelength(lambda_nm):
    """CIE 1931 2° colour-matching functions (relative scale)."""
    x = (1.056 * _g(lambda_nm, 599.8, 0.0264, 0.0323) +
         0.362 * _g(lambda_nm, 442.0, 0.0624, 0.0374) -
         0.065 * _g(lambda_nm, 501.1, 0.0490, 0.0382))
    y = (0.821 * _g(lambda_nm, 568.8, 0.0213, 0.0247) +
         0.286 * _g(lambda_nm, 530.9, 0.0613, 0.0322))
    z = (1.217 * _g(lambda_nm, 437.0, 0.0845, 0.0278) +
         0.681 * _g(lambda_nm, 459.0, 0.0385, 0.0725))
    return np.stack([x, y, z], axis=0)          # shape = (3, N)

# --- XYZ → sRGB --------------------------------------------------------------

M_XYZ2LINRGB = np.array([
    [ 3.2406, -1.5372, -0.4986],
    [-0.9689,  1.8758,  0.0415],
    [ 0.0557, -0.2040,  1.0570],
])

M_XYZ2LINP3 = np.array([
    [  2.4934969, -0.9313836, -0.4027108],
    [ -0.8294889,  1.7697341,  0.0295424],
    [  0.0358458, -0.0761724,  0.9568845],
])

def _gamma_encode(lin):
    a = 0.055
    return np.where(lin <= 0.0031308,
                    12.92 * lin,
                    (1 + a) * np.power(lin, 1/2.4) - a)

_RGB_SCALE = None      # filled lazily the first time the function is called

def wavelength_to_srgb(lambda_nm, as_bytes=True):
    """Return sRGB triplet(s) (0-1 floats or 0-255 uint8) for a wavelength."""
    xyz  = xyz_from_wavelength(lambda_nm)
    lin  = np.clip(M_XYZ2LINP3 @ xyz, 0, None)      # drop negatives

    global _RGB_SCALE
    if _RGB_SCALE is None:                           # compute once
        lam        = np.arange(380, 781)
        lin_full   = np.clip(M_XYZ2LINP3 @ xyz_from_wavelength(lam), 0, None)
        _RGB_SCALE = 1.0 / lin_full.max()            # brightest primary → 1 .0
    lin *= _RGB_SCALE
    srgb = _gamma_encode(np.clip(lin, 0, 1))

    return np.round(srgb * 255).astype(np.uint8) if as_bytes else srgb




# 2015 Data
def _gamut_map(rgb_lin):
    m = rgb_lin.max()
    return rgb_lin if m <= 1 else rgb_lin / m

cs   = colour.RGB_COLOURSPACES['sRGB']
cmfs = colour.MSDS_CMFS['CIE 2015 2 Degree Standard Observer']

def wl_to_srgb_cie2015(l_nm):
    XYZ = colour.wavelength_to_XYZ(l_nm, cmfs=cmfs)
    XYZ /= colour.wavelength_to_XYZ(555, cmfs=cmfs)[1]    
    rgb_lin = colour.XYZ_to_RGB(XYZ, cs.whitepoint,
                                cs.whitepoint, cs.matrix_XYZ_to_RGB)
    rgb_lin = _gamut_map(np.clip(rgb_lin, 0, None))
    return tuple((_gamma_encode(rgb_lin)*255 + 0.5).astype(int))







def _gauss_log(wl, a, b, c):
    """
    Log-amplitude of an asymmetric Gaussian lobe.
    wl can be scalar or ndarray.
    """
    wl = np.asarray(wl, dtype=float)
    x  = (wl - b) / c
    return a - 0.5 * x * x             # log of Gaussian, NOT exp'd yet


def _gtanh(wl, a, b, c):
    """
    Sigmoid 'generalised tanh' used by the paper.
    """
    wl = np.asarray(wl, dtype=float)
    return a / (1.0 + np.exp(c * (b - wl)))


# ---------------------------------------------------------------------------
# 5-bell fit for the CIE 1931 2° CMFs (Hwang & Park, 2025)
# ---------------------------------------------------------------------------

# S(λ)  = luminance-like “scaling” function -------------------------------
_param_S = np.array([
    ( 0.728039840518760, 452.5373300487677, 20.63998741455027),
    (-0.903768754848100, 436.5072798146079,  6.80213865708983),
    (-1.142203568915948, 426.6964514807404,  5.82897816302603),
    ( 0.418410092402545, 567.5101202778363, 43.44030592589013),
    (-0.567908922578286, 606.9866079746189, 25.57205199243601),
], dtype=float)

# chromaticity functions (sum of sigmoids) -------------------------------
_param_Y = np.array([
    ( 1.606156428203371, 502.0075537941163,  0.0967029313067696),
    (-0.800580541463583, 513.8824977974497,  0.0881332765655076),
    (-0.550389264835406, 572.0735758547771,  0.0499662323614685),
], dtype=float)

_param_Z = np.array([
    ( 0.878531722389920, 500.8587238060737, -0.1076130311623401),
    (-0.056933268287903, 470.2562688668381, -0.0756868208907367),
], dtype=float)


def _cmf_S_5b(wl):
    """
    The S(λ) bell-shaped envelope (vectorised).
    Implements the 'xi_min' guard from the C++ code to avoid underflow.
    """
    wl = np.asarray(wl, dtype=float)[..., None]      

    # log-Gaussian lobes
    xk = _gauss_log(wl, *_param_S.T)                   

    # replicate C++ underflow guard: keep terms within 36.737 of the max
    xi_min = np.max(xk, axis=-1, keepdims=True) - 36.737
    valid  = xk > xi_min

    return np.sum(np.exp(xk) * valid, axis=-1)          


def _xy_chromaticity_5b(wl):
    """
    Returns x(λ), y(λ), z(λ) **chromaticities** (they sum to 1).
    """
    wl = np.asarray(wl, dtype=float)[..., None]         

    y = (np.sum(_gtanh(wl, *_param_Y.T), axis=-1)
         + 0.007958436344230262)

    z = np.sum(_gtanh(wl, *_param_Z.T), axis=-1)

    x = 1.0 - (y + z)
    return x, y, z                                 


def xyz_from_wavelength_5b(wl):
    S   = _cmf_S_5b(wl)
    x,y,z = _xy_chromaticity_5b(wl)
    XYZ = np.stack([S*x, S*y, S*z], axis=0)          
    return XYZ

def wavelength_to_srgb_5b(lambda_nm):
    """Return sRGB triplet(s) (0-1 floats or 0-255 uint8) for a wavelength."""
    xyz  = xyz_from_wavelength_5b(lambda_nm)
    lin  = np.clip(M_XYZ2LINP3 @ xyz, 0, None)     

    global _RGB_SCALE
    if _RGB_SCALE is None:                           
        lam        = np.arange(380, 781)
        lin_full   = np.clip(M_XYZ2LINP3 @ xyz_from_wavelength_5b(lam), 0, None)
        _RGB_SCALE = 1.0 / lin_full.max()            
    lin *= _RGB_SCALE
    srgb = _gamma_encode(np.clip(lin, 0, 1))

    return np.round(srgb * 255).astype(np.uint8)








# Prepare gradients
grad_cie2015 = np.array([wl_to_srgb_cie2015(w)          for w in wavelengths], dtype=np.uint8)
grad_1931      = np.array([wavelength_to_srgb(w)        for w in wavelengths], dtype=np.uint8)
grad_spektre = np.array([wavelength_to_rgb(w)           for w in wavelengths], dtype=np.uint8)
grad_2025 = np.array([wavelength_to_srgb_5b(w)          for w in wavelengths], dtype=np.uint8)

# Prepare RGB curves
def get_rgb_curves(func):
    r = [func(w)[0] for w in wavelengths]
    g = [func(w)[1] for w in wavelengths]
    b = [func(w)[2] for w in wavelengths]
    return r, g, b

cie2015_r, cie2015_g, cie2015_b = get_rgb_curves(wl_to_srgb_cie2015)
spektre_r, spektre_g, spektre_b = get_rgb_curves(wavelength_to_rgb)
w1931_r, w1931_g, w1931_b = get_rgb_curves(wavelength_to_srgb)
w2025_r, w2025_g, w2025_b = get_rgb_curves(wavelength_to_srgb_5b)

# Plot all in a single figure
fig, axes = plt.subplots(4, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [3, 1]})

# Helper to plot curves and gradients
def plot_row(ax_row, r, g, b, grad, title, ylabel, xlabel=None):
    ax_row[0].plot(wavelengths, r, 'r', label='Red')
    ax_row[0].plot(wavelengths, g, 'g', label='Green')
    ax_row[0].plot(wavelengths, b, 'b', label='Blue')
    ax_row[0].set_title(title)
    if ylabel: ax_row[0].set_ylabel(ylabel)
    if xlabel: ax_row[0].set_xlabel(xlabel)
    ax_row[0].legend()
    ax_row[1].imshow([grad], aspect='auto', extent=[390,750,0,1])
    ax_row[1].axis('off')

plot_row(axes[0], cie2015_r, cie2015_g, cie2015_b, grad_cie2015, "CIE 2015 2° (smooth, colour-science)", "Intensity")
plot_row(axes[2], spektre_r, spektre_g, spektre_b, grad_spektre, "Spektre polynomial (original)", "Intensity", "Wavelength (nm)")
plot_row(axes[1], w1931_r, w1931_g, w1931_b, grad_1931, "CIE 1931 2° (original)", "Intensity")
plot_row(axes[3], w2025_r, w2025_g, w2025_b, grad_2025, "CIE 1931 2° (Hwang & Park, 2025)", "Intensity", "Wavelength (nm)")

plt.tight_layout()
plt.show()
