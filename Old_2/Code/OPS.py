import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_data, plot_color_seeds
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import colour
import math
from help import *
from tables import *
np.set_printoptions(legacy="1.25")
from wavelength_colors import wavelength_to_rgb, load_cmap, show_source_strip


data = {
    "angle": [129.15, 129.25, 130.03, 131.1, 131.17, 131.69, 132.0, 132.03, 132.36, 132.64, 132.74, 132.82],
    "wavelength": [404.7, 407.8, 435.8, 491.6, 496, 546.1, 577, 579.1, 623.4, 671.6, 690.7, 708.2],
    "Intensity": ['mittel-stark', 'mittel-stark', 'stark', 'mittel', 'mittel-schwach', 'stark', 'stark',          'stark',  'mittel',        'schwach', 'mittel', 'sehr schwach'],
    "colors": [   'violett',      'violett',      'blau',  'turkis', 'mint',           'grün',  'gelb(grüner)',   'gelb',   'dunkel orange', 'rot',     'rot',    'dunkel rot'],
    "angle_1":[48.37, 48.6, 49.45, 49.87, 50.03, 50.1],
    "wavelength_1": [636.6, 602.1, 518.2, 481.1, 472.2, 468],
    "intensity_1": ['stark', 'mittel-schwach', 'mittel', 'stark', 'stark', 'stark'],
    "angle_2": [130.59, 130.7, 130.77, 130.93, 131.35, 131.47, 132.2, 132.31, 132.43],
    "wavelength_2":['', 468, 472.2, 481.1, 518.2, '',  602.1, '', 636.6],
    "Intensity 2": ['schwach', 'stark', 'stark', 'stark', 'mittel', 'mittel-schwach', 'mittel-schwach', 'mittel-schwach', 'stark'],
    "colors_2": ['blau', 'blau', 'blau turkis', 'turkis', 'turkis grün', 'grün', 'orange', 'rot', 'rot'],
    "angle_3": [132.2, 132.18, 131.86, 131.17, 130.09, 129.22],
    "wavelength_3": [579.1, 577, 546.1, 491.6, 435.8, 404.7],
    "gamma":[], "gamma_2":[], "gamma_3":[], "spline_wavelength":[], "spline_wavelength_error":[], "n":[], "dn":[], "n_pow":[]
}


intensity_scale = {
    'sehr schwach': 0.1,
    'schwach': 0.25,
    'mittel-schwach': 0.4,
    'mittel': 0.6,
    'mittel-stark': 0.75,
    'stark': 1.0
}
offset = 180.8

data["gamma"] = [round(offset - angle,6) for angle in data["angle"]]
data["gamma_2"] = [round(offset - angle,6) for angle in data["angle_2"]]
data["gamma_3"] = [round(offset - angle,6) for angle in data["angle_3"]]

def sort_data(data, sorter, sorted_key, reverse=False):
    sorted_indices = np.argsort(data[sorter])
    if reverse:
        sorted_indices = sorted_indices[::-1]
    sorted_1 = [data[sorter][i] for i in sorted_indices]
    sorted_2 = [data[sorted_key][i] for i in sorted_indices]
    print("Sorted 1:", sorted_1)
    print("Sorted 2:", sorted_2)

#sort_data(data, "angle_2", "wavelength_2")

λ = np.array(data["wavelength"])

cmap, norm = load_cmap("colormaps/spectrum_wavelengths_0p01.npz")

def show_wavelength_colormap(cmap, norm):
    plt.figure(figsize=(10, 2))
    gradient = np.linspace(norm.vmin, norm.vmax, 256).reshape(1, -1)
    im = plt.imshow(gradient, aspect='auto', cmap=cmap, norm=norm, extent=[norm.vmin, norm.vmax, 0, 1])
    plt.yticks([])
    plt.colorbar(im, label='Wavelength (nm)')
    plt.title('Wavelength Color Map')
    plt.xlabel('Wavelength (nm)')
    plt.tight_layout()
    plt.show()


#show_source_strip("Data Colors", λ, [intensity_scale.get(i.lower(), None) for i in data["Intensity"]], step=0.1)
#show_source_strip("Data Colors 2", np.array(data["wavelength_1"]), [intensity_scale.get(i.lower(), None) for i in data["Intensity 2"]], step=0.1)

def generate_latex_color_definitions(data):
    """
    Generate LaTeX color definitions for each unique wavelength in the dataset.
    """
    color_definitions = []
    unique_wavelengths = set(data)
    
    for wavelength in unique_wavelengths:
        if wavelength:  # Exclude empty entries
            rgb = wavelength_to_rgb(wavelength)
            color_name = f"color_{int(wavelength)}"
            color_definitions.append(f"\\definecolor{{{color_name}}}{{RGB}}{{{rgb[0]}, {rgb[1]}, {rgb[2]}}}")
    
    return color_definitions

def apply_color_to_text(colors, wavelengths):
    """
    Modify the 'colors' field in data to include LaTeX color commands.
    """
    colored_texts = []
    for i, color_text in enumerate(colors):
        wavelength = wavelengths[i]
        
        if wavelength == '' or math.isnan(wavelength):
            colored_texts.append(color_text)
            continue
        color_name = f"color_{int(wavelength)}"  # Match color definition name
        colored_texts.append(f"\\textcolor{{{color_name}}}{{{color_text}}}")

    return colored_texts  # Replace original color texts with colorized versions

#color_definitions = generate_latex_color_definitions(data["wavelength"])
#for color_def in color_definitions:
#    print(color_def)
data["colors"] = apply_color_to_text(data["colors"], data["wavelength"])

headers = {
    "angle": {"label": "{Position(°)}", "precision": 2, "err": [0.01]*len(data["angle"])},      
    "gamma": {"label": "{rel.Pos. $\\gamma$(°)}", "precision": 2, "err": [0.02]*len(data["gamma"])},
    "colors": {"label": "{Farbe}", "dark": True},
    "Intesity": {"label": "{Intensität}"},
    "wavelength": {"label": "{$\\lambda$(nm)}"}
}

print_standard_table(
    data=data,
    headers=headers,
    column_formats= ["3.2", "3.2", "3.0", "4.0", "2.1"],
    caption="Messdaten einer Hg-Lampe mit den zugeordneten Farben und Intensitäten der charakteristischen Spektrallinien. Die Positionen der Linien sind relativ zum Referenzwinkel $\\gamma_0$ als $\\gamma$ angegeben, und die Wellenlängen sind mit den entsprechenden Farben und Intensitäten aufgeführt.",
    label="tab:HgSpektrum",
    show=False
)


def monte_carlo_spline_uncertainty(
    x, y, xerr=None, y_error=None, 
    spline_kwargs=None, n_simulations=1000, 
    x_smooth=None, confidence=95):

    x = np.array(x)
    y = np.array(y)
    
    if spline_kwargs is None:
        spline_kwargs = {'s': 0, 'k': 3} 
    
    if x_smooth is None:
        x_smooth = np.linspace(np.min(x), np.max(x), 200)
    
    spline_predictions, spline_derivatives = np.zeros((n_simulations, len(x_smooth))), np.zeros((n_simulations, len(x_smooth)))
    
    # Monte Carlo simulation
    if xerr is not None:
        x_simulated = np.array([x + np.random.normal(0, xerr, size=len(x)) for _ in range(n_simulations)])
    else:
        x_simulated = np.tile(x, (n_simulations, 1))
 
    if y_error is not None:
        y_simulated = np.array([y + np.random.normal(0, y_error, size=len(y)) for _ in range(n_simulations)])
    else:
        y_simulated = np.tile(y, (n_simulations, 1))

    # Ensure that simulated data is within the valid range
    x_simulated = np.clip(x_simulated, np.min(x), np.max(x))
    y_simulated = np.clip(y_simulated, np.min(y), np.max(y))

    for i in range(n_simulations):
        idx = np.argsort(x_simulated[i])
        x_sim = x_simulated[i][idx].astype(float)
        y_sim = y_simulated[i][idx].astype(float)

        # collapse duplicate x by averaging y
        ux, inv = np.unique(np.round(x_sim, 12), return_inverse=True)
        y_acc = np.zeros_like(ux, dtype=float)
        n_acc = np.zeros_like(ux, dtype=int)
        np.add.at(y_acc, inv, y_sim)
        np.add.at(n_acc, inv, 1)
        x_sim, y_sim = ux, y_acc / n_acc

        # need at least k+1 unique points
        if x_sim.size <= spline_kwargs.get('k', 3):
            spline_predictions[i, :] = np.nan
            spline_derivatives[i, :] = np.nan
            continue

        try:
            spline_sim = UnivariateSpline(x_sim, y_sim, **spline_kwargs)
            spline_predictions[i, :] = spline_sim(x_smooth)
            spline_derivatives[i, :] = spline_sim.derivative()(x_smooth)
        except Exception as e:
            print(f"Simulation {i} failed: {e}")
            spline_predictions[i, :] = np.nan
            spline_derivatives[i, :] = np.nan
    
    # Remove simulations that failed (contain NaN values)
    valid_indices = ~np.isnan(spline_predictions).any(axis=1)
    spline_predictions = spline_predictions[valid_indices]
    spline_derivatives = spline_derivatives[valid_indices]
    
    # Compute mean and confidence intervals for spline predictions
    median_spline     = np.nanmedian(spline_predictions, axis=0)
    median_derivative = np.nanmedian(spline_derivatives, axis=0)


    bounds = lambda data: (np.nanpercentile(data, (100 - confidence) / 2, axis=0), 
                           np.nanpercentile(data, 100 - (100 - confidence) / 2, axis=0))
    lower_bound_spline, upper_bound_spline = bounds(spline_predictions)
    lower_bound_spline_derivative, upper_bound_spline_derivative = bounds(spline_derivatives)
    
    return (median_spline, lower_bound_spline, upper_bound_spline,
            median_derivative, lower_bound_spline_derivative, upper_bound_spline_derivative,
            spline_predictions, spline_derivatives)





data_val = sorted(zip(data["gamma"], data["wavelength"]))
x = [point[0] for point in data_val]
y = [point[1] for point in data_val]

dx = 0.02

xerr = np.full_like(x, dx)

x2 = sorted(data["gamma_2"])

spline = UnivariateSpline(x, y, s=200, k=4) 
angle_smooth = np.linspace(min(x), max(x), 1000)


(median_spline, lower_bound, upper_bound,
 median_derivative, lower_bound_derivative, upper_bound_derivative,
 spline_predictions, spline_derivatives) = monte_carlo_spline_uncertainty(
    x, y, xerr=xerr, y_error=None, 
    spline_kwargs={'s':200, 'k':4}, 
    n_simulations=3000, x_smooth=angle_smooth, confidence=95)





plot_data(
    filename="Plots/OPS_1.pdf",
    datasets=[
        {
            "xdata": x,
            "ydata": y,
            "xerr": dx,
            "label": "Hg Spektrum",
            "marker": ".",
            "color_group": "1",
            "fit_xdata": angle_smooth,
            "fit": spline(angle_smooth),
            "confidence": [(None, None), (lower_bound, upper_bound)],
        },
        {
            "xdata": x2,
            "ydata": spline(x2),
            "marker": "*",
            "xerr": dx,
            "label":"Unbekannte Lampe mit λ von Kalibrierkurve"
        },
        {
            "xdata": data["angle_1"],
            "ydata": data["wavelength_1"],
            "marker": "x",
            "xerr": dx,
            "label":"Zn Spektrum mit Winkeln von Unbekannter Lampe"
        },
        {
            "xdata": [48.27],
            "ydata": [636.6],
            "marker": "x",
            "xerr": dx,
            "label": "Zn 646.1nm mit Originalwinkel(Ohne Verschiebung) ",
            "color": "red"
        },
    ],
    xlabel="relativer Ablenkwinkel γ/°", 
    ylabel="Wellenlänge λ/nm",
    title="Kalibrierkurve (Spline) des Hg-Spektrums mit Zink-Spektrum",
    color_seed=84,
    plot=False
)

# plot the residuals of the data to spline fit
plot_data(
    filename="Plots/OPS_2.pdf",
    datasets=[
        {
            "xdata": x,
            "ydata": y - spline(x),
            "xerr": dx,
            "label": "Hg Spektrum Residuen",
            "marker": ".",
            "color_group": "1"
        },
        {
            "xdata": x,
            "ydata": np.zeros_like(x),
            "fit_xdata": angle_smooth,
            "marker": "None",
            "line": "--",
            "label": "Spline Fit Kalibrierkurve",
            "confidence": [(None, None), (lower_bound - spline(angle_smooth), upper_bound - spline(angle_smooth))],
            "color_group": "1"
        },
        {
            "xdata": x2,
            "ydata": spline(x2) - spline(x2),
            "marker": "*",
            "xerr": dx,
            "label":"Unbekannte Lampe mit λ von Kalibrierkurve"
        },
        {
            "xdata": data["angle_1"],
            "ydata": data["wavelength_1"] - spline(data["angle_1"]),
            "marker": "x",
            "xerr": dx,
            "label":"Zn Spektrum"
        },
    ],
    xlabel="relativer Ablenkwinkel γ/°", 
    ylabel="Wellenlänge λ/nm",
    title="Residuen des Spline Fits",
    color_seed=84,
    height = 10,
    plot=False
)


data["spline_wavelength"] = [round(wavelength, 1) for wavelength in spline(data["gamma_2"])]

def find_closest(xdata):
    return np.argmin(np.abs(xdata - angle_smooth))


interval = [(upper_bound[find_closest(angle)] - lower_bound[find_closest(angle)]) / 2 for angle in data["gamma_2"]]

data["spline_wavelength_error"] = [dwavelength for dwavelength in interval]

#color_definitions = generate_latex_color_definitions(data["wavelength_2"])
#for color_def in color_definitions:
#    print(color_def)
data["colors_2"] = apply_color_to_text(data["colors_2"], data["wavelength_2"])


headers_2 = {
    "angle_2": {"label": "{Position(°)}", "precision": 2, "err": [0.01]*len(data["angle_2"])},
    "gamma_2": {"label": "{rel.Pos. $\\gamma$(°)}", "precision": 2, "err": [dx]*len(data["gamma_2"])},
    "colors_2": {"label": "{Farbe}"},
    "Intensity 2": {"label": "{Intensität}"},
    "spline_wavelength": {"label": "{$\\lambda_{Spline}$(nm)}","precision": 1, "err": data["spline_wavelength_error"] },
    "wavelength_2": {"label": "{$\\lambda_{Zn}$(nm)}"}
}


print_standard_table(
    data=data,
    headers=headers_2,
    column_formats= ["3.2", "3.2", "3.0", "4.0", "2.1", "2.1"],
    caption="Messdaten einer unbekannten Lampe 6. Die Wellenlängen wurden mithilfe des Zink-Spektrums zugeordnet. Einige Linien konnten keiner spezifischen Wellenlänge zugeordnet werden. Die Spalte $\\lambda_{Spline}$ zeigt die Wellenlängen, die durch die Kalibrierkurve bestimmt wurden, während die Spalte $\\lambda_{Zn}$ die zugeordneten Wellenlängen des Zink-Spektrums darstellt.",
    label="tab:ZnSpektrum",
    show=False
)


data_val_2 = sorted(zip(data["gamma_3"], data["wavelength_3"]))
x3 = [point[0] for point in data_val_2]
y3 = [point[1] for point in data_val_2]

n3_0 = [ 2* np.sin(np.radians(angle/2 +30)) for angle in x3]
y3, n3 = zip(*sorted(zip(y3, n3_0)))
y3 = np.array(y3) * 1e-3

dn_3 = [
   np.cos(np.radians(angle/2 +30)) * (dx * np.pi / 180)
    for angle in x3
]

spline_3 = UnivariateSpline(y3, n3, s=700, k=3)
wavelength_smooth_3 = np.linspace(min(y3), max(y3), 1000)
n3_spline = [spline_3(wavelength) for wavelength in wavelength_smooth_3]

(median_spline_3, lower_bound, upper_bound, 
 median_derivative_3, lower_bound_derivative, upper_bound_derivative,
 spline_predictions, spline_derivatives) = monte_carlo_spline_uncertainty(
    y3, n3, y_error=dn_3,
    spline_kwargs={'s':700, 'k':3}, 
    n_simulations=2000, x_smooth=wavelength_smooth_3, confidence=95)


def refractive_index_polynomial(wavelength, a0=2.5730565, a1=-0.010203605, a2=0.023095784, a3=0.00087114252, a4=-3.0730079e-5, a5=6.9457986e-6):
    n_squared = (
        a0 +
        a1 * wavelength**2 +
        a2 * wavelength**-2 +
        a3 * wavelength**-4 +
        a4 * wavelength**-6 +
        a5 * wavelength**-8
    )
    n_squared = np.maximum(n_squared, 0)
    return np.sqrt(n_squared)

initial_guess = [1.34533359, 0.00997743871, ] # 0.209073176, 0.0470450767]  0.937357162, 111.886764

def sellmeier_eq(wavelength, B1, C1):
    term1 = (B1 * wavelength**2) / (wavelength**2 - C1)
    return np.sqrt(1 + term1)


popt, pcov = curve_fit(sellmeier_eq, y3, n3, p0=initial_guess, sigma=dn_3, absolute_sigma=True)
n_fit = sellmeier_eq(wavelength_smooth_3, *popt)

def dn_dx(x, B, C):
 return -((B * C * x) / ((C - x**2)**2 * np.sqrt(1 + B + (B * C) / (-C + x**2))))

def dn_dB(wavelength, B, C):
    term1 = -C + wavelength**2
    term2 = 1 + (B * wavelength**2) / term1
    sqrt_term = np.sqrt(term2)
    denominator = 2 * term1 * sqrt_term
    numerator = wavelength**2
    return numerator / denominator

def dn_dC(wavelength, B, C):
    term1 = -C + wavelength**2
    term2 = 1 + (B * wavelength**2) / term1
    sqrt_term = np.sqrt(term2)
    denominator = 2 * (term1)**2 * sqrt_term
    numerator = B * wavelength**2
    return numerator / denominator

def ddn_dx_dC(x, B, C):
    return (B*x*(-B*C * x**2 - 2*(B + 1)*x**4 + 2*C**2))/(2*(C - x**2)**4 * ((B*C)/(x**2 - C) + B + 1)**(3/2))

def ddn_dx_dB(x, B, C):
    return (C*x*(-2*C + (2 + B)*x**2))/(2*(C - x**2)**3*(1 + B + (B*C)/(-C + x**2))**(3/2))



dn3_sellmeier = np.sqrt( dn_dB(wavelength_smooth_3, *popt)**2 * pcov[0, 0] + dn_dC(wavelength_smooth_3, *popt)**2 * pcov[1, 1] + 2 * dn_dC(wavelength_smooth_3, *popt)*dn_dB(wavelength_smooth_3, *popt) * pcov[0, 1] )

dn_dx_values = dn_dx(wavelength_smooth_3, *popt)

def dn_dxerr(wavelength):
 return np.sqrt(
        (ddn_dx_dB(wavelength, *popt) ** 2) * pcov[0, 0] +
        (ddn_dx_dC(wavelength, *popt) ** 2) * pcov[1, 1] +
        2 * ddn_dx_dB(wavelength, *popt) * ddn_dx_dC(wavelength, *popt) * pcov[0, 1]
    )


perr = np.sqrt(np.diag(pcov))
print(pcov)
print("B1, C1:", popt)
print("Errors ΔB1, ΔC1:", perr)

def find_closest(wavelength):
    return np.argmin(np.abs(wavelength - wavelength_smooth_3))



def uncertainty_spline(x):
    return ( UnivariateSpline(wavelength_smooth_3, upper_bound, s=10)(x) - UnivariateSpline(wavelength_smooth_3, lower_bound, s=10)(x)) / 2

def uncertainty_sellmeier(x):
    return ( UnivariateSpline(wavelength_smooth_3, n_fit + 2 * dn3_sellmeier, s=10)(x) - UnivariateSpline(wavelength_smooth_3, n_fit - 2 * dn3_sellmeier, s=10)(x))/2

def compute_derivative_at_point(f, x0, initial_h=0.1, N=100, params=None, max_iterations=50, tolerance=1e-6, h_min=1e-8, yerr_func=None):
    h = initial_h
    prev_slope = None
    for i in range(max_iterations):
        xdata = np.linspace(x0 - h, x0 + h, N)
        if params is None:
            ydata = f(xdata)
        else:
            ydata = f(xdata, *params)
        
        if yerr_func is not None:
            yerr = yerr_func(xdata)

        try:
            a, da, b, db, R2, s2 = slope(xdata, ydata, yerr)
        except ValueError as e:
            print(f"Iteration {i}: {e}")
            h = h * 2  # Increase h to retry
            if h > initial_h:
                break
            continue

        if prev_slope is not None:
            if abs(b - prev_slope) < tolerance * abs(prev_slope):
                break
        prev_slope = b
        h = h / 2
        if h < h_min or b == 0:
            h = h * 2  # Increase h to retry
            if h > initial_h:
                break
            continue
    return (a, da), (b, db), R2, s2

a_spline_579, b_spline_579, R2_spline_579, s2_spline_579 = compute_derivative_at_point(spline_3, 0.5791, yerr_func=uncertainty_spline)
a_spline_577, b_spline_577, R2_spline_577, s2_spline_577 = compute_derivative_at_point(spline_3, 0.577, yerr_func=uncertainty_spline)
a_sellmeier_577, b_sellmeier_577, R2_sellmeier_577, s2_sellmeier_577 = compute_derivative_at_point(sellmeier_eq, 0.577, params=popt, yerr_func=uncertainty_sellmeier)
a_sellmeier_579, b_sellmeier_579, R2_sellmeier_579, s2_sellmeier_579 = compute_derivative_at_point(sellmeier_eq, 0.5791, params=popt, yerr_func=uncertainty_sellmeier)




plot_data(
    filename="Plots/OPS_4.pdf",
    datasets=[
        {
            "xdata": y3,
            "ydata": n3,
            "yerr": dn_3,
            "label": "n(λ)",
            "marker": ".",
            "color_group": "1",
            "fit_xdata": wavelength_smooth_3,
            "fit": n3_spline,
            "confidence": [(None, None), (lower_bound, upper_bound)],
        },
        {
            "xdata": wavelength_smooth_3,
            "ydata": n_fit,
            "label": "Sellmeier Fit",
            "marker": "None",
            "line": "-",
            "color_group": "2",
            "confidence": [(None, None), (n_fit - 2 * dn3_sellmeier, n_fit + 2 * dn3_sellmeier)],
        },
        {
            "xdata": wavelength_smooth_3,
            "ydata": b_spline_579[0]*wavelength_smooth_3 + a_spline_579[0],
            "marker": "None",
            "line": "--",
            "color_group": "1",
            "label": "Tangenten von Spline"
        },
        {
            "xdata": wavelength_smooth_3,
            "ydata": b_spline_577[0]*wavelength_smooth_3 + a_spline_577[0],
            "marker": "None",
            "line": "--",
            "color_group": "1",
        },
        {
            "xdata": wavelength_smooth_3,
            "ydata": b_sellmeier_577[0]*wavelength_smooth_3 + a_sellmeier_577[0],
            "marker": "None",
            "line": "--",
            "color_group": "2",
            "label": "Tangenten von Sellmeier Fit"
        },
        {
            "xdata": wavelength_smooth_3,
            "ydata": b_sellmeier_579[0]*wavelength_smooth_3 + a_sellmeier_579[0],
            "marker": "None",
            "line": "--",
            "color_group": "2",
        },
    ],
    xlabel="Wellenlänge λ/µm",
    ylabel="Brechungsindex n",
    title="Brechungsindex n in Abhängigkeit der Wellenlänge λ",
    color_seed=77,
    plot=False
)

plot_data(
    filename="Plots/OPS_4_1.pdf",
    datasets=[
        {
            "xdata": wavelength_smooth_3,
            "ydata": [0] * len(wavelength_smooth_3),
            "label": "Spline Fit",
            "marker": "None",
            "line": "-",
            "color_group": "1",
            "confidence": [(None, None), (lower_bound - n3_spline, upper_bound - n3_spline)],
        },
        {
            "xdata": y3,
            "ydata": n3 - spline_3(y3),
            "yerr": dn_3,
            "label": "n(λ)",
            "marker": ".",
            "line": "None",
            "color": "black"
        },
    ],
    xlabel="relativer Ablenkwinkel γ/°", 
    ylabel="Wellenlänge λ/nm",
    title="Residuen des Spline Fits",
    height = 10,
    color_seed=77,
    plot=False
)

plot_data(
    filename="Plots/OPS_4_2.pdf",
    datasets=[
        {   
            "xdata":[],
            "ydata":[],
            "color_group": "1"
        },
        {
            "xdata": wavelength_smooth_3,
            "ydata": [0] * len(wavelength_smooth_3),
            "label": "Sellmeier Fit",
            "marker": "None",
            "line": "-",
            "color_group": "2",
            "confidence": [(None, None), (-2*dn3_sellmeier, 2*dn3_sellmeier)],
        },
        {
            "xdata": y3,
            "ydata": n3 - sellmeier_eq(y3, *popt),
            "yerr": dn_3,
            "label": "n(λ)",
            "marker": ".",
            "line": "None",
            "color": "black"
        },

    ],
    xlabel="relativer Ablenkwinkel γ/°", 
    ylabel="Wellenlänge λ/nm",
    title="Residuen des Sellmeier Fits",
    height = 10,
    color_seed=77,
    plot=False
)




data["n"], data["dn"], data["n_pow"] = map(list, zip(*(round_val(n, dn) for n, dn in zip(n3[::-1], dn_3[::-1]))))

data["wavelength_3"] = apply_color_to_text(data["wavelength_3"], data["wavelength_3"])

headers_3 = {
    "angle_3": {"label": "{Position(°)}", "precision": 2, "err": [0.01]*len(data["angle_3"])},      
    "gamma_3": {"label": "{rel.Pos. $\\gamma$(°)}", "precision": 2, "err": [dx]*len(data["gamma_3"])},
    "n": {"label": "{n $\\pm \\Delta n$}", "precision": data["n_pow"], "err": data["dn"]},                 
    "wavelength_3": {"label": "{$\\lambda$(nm)}", "dark": True},
}

print_standard_table(
    data=data,
    headers=headers_3,
    column_formats= ["3.2", "3.2", "1.4", "1.4"],
    caption="Messwerte des Hg-Spektrums, wobei alle Winkel die minimale Ablenkung für jede Wellenlänge im Prisma darstellen.",
    label="tab:HgMinima",
    show=False
)



def find_closest(wavelength):
    return np.argmin(np.abs(wavelength - wavelength_smooth_3))

dn_dlambda_spline_577_1 = median_derivative_3[find_closest(577 * 1e-3)]
dn_dlambda_spline_579_1 = median_derivative_3[find_closest(579.1 * 1e-3)]
dn_dlambda_577_1 = dn_dx(577*1e-3, *popt)
dn_dlambda_579_1 = dn_dx(579*1e-3, *popt)

low_bound_n = [dn_dx(wavelength, *popt) for wavelength in wavelength_smooth_3] - 2 * dn_dxerr(wavelength_smooth_3)
high_bound_n = [dn_dx(wavelength, *popt) for wavelength in wavelength_smooth_3] + 2 * dn_dxerr(wavelength_smooth_3)



def find_ddn_dx_spline(wavelength):
    low_der = lower_bound_derivative[find_closest(wavelength)] - dn_dlambda_spline_577_1
    up_der = upper_bound_derivative[find_closest(wavelength)] - dn_dlambda_spline_579_1
    return low_der, up_der

dn_dlambda_577, ddn_dlambda_577, _ = round_val( dn_dlambda_577_1, err= 2 * dn_dxerr(0.577) ,intermed=False)
dn_dlambda_579, ddn_dlambda_579, _ = round_val( dn_dlambda_579_1, err = 2* dn_dxerr(0.5791) ,intermed=False)

dn_dlambda_spline_577, ddn_dlambda_spline_577, _ = round_val( float(dn_dlambda_spline_577_1), err = (abs(find_ddn_dx_spline(0.577)[0]) + abs(find_ddn_dx_spline(0.577)[1])) / 2, intermed=False)
dn_dlambda_spline_579, ddn_dlambda_spline_579, _ = round_val( float(dn_dlambda_spline_579_1), err = (abs(find_ddn_dx_spline(0.5791)[0]) + abs(find_ddn_dx_spline(0.5791)[1])) / 2, intermed=False)

plot_data(
    filename="Plots/OPS_5.pdf",
    datasets=[
        {
            "ydata": [dn_dlambda_spline_577_1, dn_dlambda_spline_579_1],
            "xdata": [0.577, 0.5791],
            "yerr": [(abs(find_ddn_dx_spline(0.577)[0]) + abs(find_ddn_dx_spline(0.577)[1]))/2, (abs(find_ddn_dx_spline(0.5791)[0]) + abs(find_ddn_dx_spline(0.5791)[1]))/2],
            "marker":".",
            "fit": median_derivative_3,
            "fit_xdata": wavelength_smooth_3,
            "confidence": [(None, None), (lower_bound_derivative, upper_bound_derivative)],
            "label":"Ableitung Spline Fit an 577nm und 579nm",
            "color_group": "1",
        },
        {
            "ydata": [dn_dlambda_577_1, dn_dlambda_579_1],
            "xdata": [0.577, 0.5791],
            "yerr": [2 * dn_dxerr(0.577), 2 * dn_dxerr(0.5791)],
            "fit": [dn_dx(wavelength, *popt) for wavelength in wavelength_smooth_3],
            "fit_xdata": wavelength_smooth_3,
            "confidence": [(None, None), (low_bound_n, high_bound_n)],
            "marker":".",
            "label":"Ableitung Sellmeier Fit an 577nm und 579nm",
            "color_group": "2",
        },
    ],
    xlabel="Wellenlänge λ/µm",
    ylabel="dn/dλ",
    title="Ableitungen der Brechungsindizes",
    color_seed=77,
    plot=False
)

#with tangents at 577nm and 579nm
print("tanget at 577nm and 579nm")
print(f"dn/d\\lambda(577 \\micro m)_{{sellmeier}} =( {round_val(b_sellmeier_577[0],b_sellmeier_577[1], intermed=False)[0]} \\pm {round_val(b_sellmeier_577[0],b_sellmeier_577[1], intermed=False)[1]}) (\\micro m)^{{-1}}, R^2={R2_sellmeier_577:.2f}, \\sigma^2= {round_val(s2_sellmeier_577, intermed=False)[0]}| dn/d\\lambda(579.1 \\micro m)_{{sellmeier}}= ({round_val(b_sellmeier_579[0],b_sellmeier_579[1], intermed=False)[0]} \\pm {round_val(b_sellmeier_579[0],b_sellmeier_579[1], intermed=False)[1]})(\\micro m)^{{-1}}, R^2={R2_sellmeier_579:.2f}, \\sigma^2= {round_val(s2_sellmeier_579, intermed=False)[0]}" )
print(f"dn/d\\lambda(577 \\micro m)_{{spline}} = ({round_val(b_spline_577[0],b_spline_577[1], intermed=False)[0]} \\pm {round_val(b_spline_577[0],b_spline_577[1], intermed=False)[1]})(\\micro m)^{{-1}}, R^2={R2_spline_577:.2f}, \\sigma^2= {round_val(s2_spline_577, intermed=False)[0]}| dn/d\\lambda(579.1 \\micro m)_{{spline}}= ({round_val(b_spline_579[0],b_spline_579[1], intermed=False)[0]} \\pm {round_val(b_spline_579[0],b_spline_579[1], intermed=False)[1]})(\\micro m)^{{-1}}, R^2={R2_spline_579:.2f}, \\sigma^2= {round_val(s2_spline_579, intermed=False)[0]}" )



print("derivatives at 577nm and 579nm")
print(f"dn/d\\lambda(0.577 \\micro m)_{{sellmeier}} = ({dn_dlambda_577} \\pm {ddn_dlambda_577})(\\micro m)^{{-1}}| dn/d\\lambda(0.579 \\micro m)_{{sellmeier}} = ({dn_dlambda_579} \\pm {ddn_dlambda_579})(\\micro m)^{{-1}}" )
print(f"dn/d\\lambda(0.577 \\micro m)_{{spline}} = ({dn_dlambda_spline_577} \\pm {ddn_dlambda_spline_577})(\\micro m)^{{-1}} | dn/d\\lambda(0.579 \\micro m)_{{spline}} = ({dn_dlambda_spline_579} \\pm {ddn_dlambda_spline_579})(\\micro m)^{{-1}}" )


resolve_theo = 0.5 * (577 + 579.1) / (579.1 - 577)
resolve_exp_577 =  2.04 * 1e-3 / np.cos(np.radians( (data["gamma_3"][1]+60)/2) ) * abs(dn_dlambda_577) *1e6
resolve_exp_579 =  2.04 * 1e-3 / np.cos(np.radians( (data["gamma_3"][0]+60)/2) ) * abs(dn_dlambda_579) *1e6

# resolve lambda/dlambda = d/cos(gamma/2 + 30°) *dn/dlambda 
# dresolve = sqrt((d(lambda/dlambda)/dd)^2 *deltad^^2 + (d(lambda/dlambda)/dgamma)^2 *delta(gamma)^2 + (d(lambda/dlambda)/ d(dn/dlambda))^2 *delta(dn/dlambda)^2)
dresolve_577 = np.sqrt((resolve_exp_577 * np.tan(np.radians( (data["gamma_3"][1]+60)/2) ) * 0.02*np.pi/180)**2 + (resolve_exp_577/abs(dn_dlambda_577)*1e-6 * ddn_dlambda_577)**2)
dresolve_579 = np.sqrt((resolve_exp_579 * np.tan(np.radians( (data["gamma_3"][0]+60)/2) ) * 0.02*np.pi/180)**2 + (resolve_exp_579/abs(dn_dlambda_579)*1e-6 * ddn_dlambda_579)**2)


print({round_val(resolve_theo, intermed=False)[0]})
print(f" {round_val(resolve_exp_577, err=dresolve_577, intermed=False)[0]} \\pm {round_val(resolve_exp_577, err=dresolve_577, intermed=False)[1]}, {round_val(resolve_exp_579, err=dresolve_579, intermed=False)[0]} \\pm {round_val(resolve_exp_579, err=dresolve_579, intermed=False)[1]}")

resolve_exp_1 =  2.04 * 1e-3 / np.cos(np.radians( (data["gamma_3"][1]+60)/2) ) * abs(dn_dlambda_spline_577) *1e6
resolve_exp_2 =  2.04 * 1e-3 / np.cos(np.radians( (data["gamma_3"][0]+60)/2) ) * abs(dn_dlambda_spline_579) *1e6

print(resolve_exp_1, resolve_exp_2)

