import wave
import numpy as np
from plotting_minus import plot_data
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import math
from help import *
from scipy.optimize import curve_fit
from tables import *
from scipy.optimize import fsolve

data = {
    "angle": [129.15, 129.25, 130.03, 131.1, 131.17, 131.69, 132.0, 132.03, 132.36, 132.64, 132.74, 132.82],
    "wavelength": [404.7, 407.8, 435.8, 491.6, 496, 546.1, 577, 579.1, 623.4, 671.6, 690.7, 708.2],
    "Intesity": ['mittel-stark', 'mittel-stark', 'stark', 'mittel', 'mittel-schwach', 'stark', 'stark', 'stark', 'mittel', 'schwach', 'mittel', 'sehr schwach'],
    "colors": ['violett', 'violett', 'blau', 'turkis', 'mint', 'grün', 'gelb(grüner)', 'gelb', 'dunkel orange', 'rot', 'rot', 'dunkel rot'],
    "angle_1":[48.37, 48.6, 49.45, 49.87, 50.03, 50.1],
    "wavelength_1": [636.6, 602.1, 518.2, 481.1, 472.2, 468],
    "angle_2": [130.59, 130.7, 130.77, 130.93, 131.35, 131.47, 132.2, 132.31, 132.43],
    "wavelength_2":['', 468, 472.2, 481.1, 518.2, '',  602.1, '', 636.6],
    "Intensity 2": ['schwach', 'stark', 'stark', 'stark', 'mittel', 'mittel schwach', 'mittel schwach', 'mittel schwach', 'stark'],
    "colors_2": ['blau', 'blau', 'blau turkis', 'turkis', 'turkis grün', 'grün', 'orange', 'rot', 'rot'],
    "angle_3": [132.2, 132.18, 131.86, 131.17, 130.09, 129.22],
    "wavelength_3": [579.1, 577, 546.1, 491.6, 435.8, 404.7],
    "gamma":[], "gamma_2":[], "gamma_3":[], "spline_wavelength":[], "n":[], "dn":[], "n_pow":[]
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

def wavelength_to_rgb(wavelength):
    # Return black for wavelengths outside the visible range
    if wavelength < 380 or wavelength > 780 or math.isnan(wavelength):
        return (0, 0, 0)

    # Define RGB components based on wavelength ranges
    if 380 <= wavelength < 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength < 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength < 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wavelength < 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif 645 <= wavelength <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = G = B = 0

    # Adjust intensity for wavelengths at the edges of the visible spectrum
    if 380 <= wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 645 <= wavelength <= 780:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 645)
    else:
        factor = 1.0

    R = int(np.clip(R * factor * 255, 0, 255))
    G = int(np.clip(G * factor * 255, 0, 255))
    B = int(np.clip(B * factor * 255, 0, 255))

    return (R, G, B)

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
    "angle": {"label": "{Position(°)}", "precision": 2},      
    "gamma": {"label": "{rel.Pos. $\\gamma$(°)}", "precision": 2},
    "colors": {"label": "{Farbe}"},
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





data_val = sorted(zip(data["gamma"], data["wavelength"]))
x = [point[0] for point in data_val]
y = [point[1] for point in data_val]

x_error = np.full_like(x, 0.02)

x2 = sorted(data["gamma_2"])


spline = UnivariateSpline(x, y, s=100, k=4) 
# Generate points for plotting the spline fit
angle_smooth = np.linspace(min(x), max(x), 1000)
wavelength_smooth = spline(angle_smooth)



def monte_carlo_spline_uncertainty(
    x, y, x_error=None, y_error=None, 
    spline_kwargs=None, n_simulations=1000, 
    x_smooth=None, confidence=95):
    # Ensure x and y are NumPy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Set default spline parameters if not provided
    if spline_kwargs is None:
        spline_kwargs = {'s': 0, 'k': 3}  # Default to cubic spline with no smoothing
    
    # Create x_smooth if not provided
    if x_smooth is None:
        x_smooth = np.linspace(np.min(x), np.max(x), 200)
    
    # Initialize arrays to store spline predictions and derivatives
    spline_predictions = np.zeros((n_simulations, len(x_smooth)))
    spline_derivatives = np.zeros((n_simulations, len(x_smooth)))
    
    # Monte Carlo simulation
    for i in range(n_simulations):
        # Simulate x-values if x_error is provided
        if x_error is not None:
            x_simulated = x + np.random.normal(0, x_error, size=len(x))
        else:
            x_simulated = x.copy()
        
        # Simulate y-values if y_error is provided
        if y_error is not None:
            y_simulated = y + np.random.normal(0, y_error, size=len(y))
        else:
            y_simulated = y.copy()
        
        # Sort x_simulated and corresponding y_simulated to maintain order
        sorted_indices = np.argsort(x_simulated)
        x_sim = x_simulated[sorted_indices]
        y_sim = y_simulated[sorted_indices]
        
        # Fit spline to simulated data
        try:
            spline_sim = UnivariateSpline(x_sim, y_sim, **spline_kwargs)
            # Evaluate spline on the grid
            spline_predictions[i, :] = spline_sim(x_smooth)
            # Evaluate derivative of spline on the grid
            spline_derivatives[i, :] = spline_sim.derivative()(x_smooth)
        except Exception as e:
            print(f"Simulation {i} failed: {e}")
            # Handle exceptions or set predictions to NaN
            spline_predictions[i, :] = np.nan
            spline_derivatives[i, :] = np.nan
    
    # Remove simulations that failed (contain NaN values)
    valid_indices = ~np.isnan(spline_predictions).any(axis=1)
    spline_predictions = spline_predictions[valid_indices]
    spline_derivatives = spline_derivatives[valid_indices]
    
    # Compute mean and confidence intervals for spline predictions
    mean_spline = np.nanmean(spline_predictions, axis=0)
    lower_percentile = (100 - confidence) / 2
    upper_percentile = 100 - lower_percentile
    lower_bound = np.nanpercentile(spline_predictions, lower_percentile, axis=0)
    upper_bound = np.nanpercentile(spline_predictions, upper_percentile, axis=0)
    
    # Compute mean and confidence intervals for spline derivatives
    mean_derivative = np.nanmean(spline_derivatives, axis=0)
    lower_bound_derivative = np.nanpercentile(spline_derivatives, lower_percentile, axis=0)
    upper_bound_derivative = np.nanpercentile(spline_derivatives, upper_percentile, axis=0)
    
    return (mean_spline, lower_bound, upper_bound, 
            mean_derivative, lower_bound_derivative, upper_bound_derivative,
            spline_predictions, spline_derivatives)

(mean_spline, lower_bound, upper_bound, 
 mean_derivative, lower_bound_derivative, upper_bound_derivative,
 spline_predictions, spline_derivatives) = monte_carlo_spline_uncertainty(
    x, y, x_error=x_error, y_error=None, 
    spline_kwargs={'s':100, 'k':4}, 
    n_simulations=1000, x_smooth=angle_smooth, confidence=95)



Zn_spectrum  = [468, 471.3, 472.2, 481.1, 518.2, 589.4, 602.1, 636.6, 648.2, 692.8, 693.8]

def find_x_for_y(y_target, spline, initial_guess):
    return fsolve(lambda x: spline(x) - y_target, x0=initial_guess)[0]

initial_guesses = [find_x_for_y(y, spline, initial_guess) for y, initial_guess in zip(Zn_spectrum, [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6])]


plot_data(
    filename="Plots/OPS_1.pdf",
    datasets=[
        {
            "xdata": x,
            "ydata": y,
            "x_error": [0.02] ,
            "label": "Hg Spektrum",
            "marker": ".",

        },
        {
            "xdata": angle_smooth,
            "ydata": wavelength_smooth,
            "marker": "None",
            "label": "Spline Fit Kalibrierkurve durch Hg Spektrum",
        },
        {
            "xdata": angle_smooth,
            "confidence": True,
            "low_bound": lower_bound,
            "high_bound": upper_bound,
            "label": "95% Konfidenzintervall",
        },
        {
            "xdata": x2,
            "ydata": spline(x2),
            "marker": "*",
            "x_error": [0.02],
            "label":"Unbekannte Lampe mit λ von Kalibrierkurve"
        },
        {
            "xdata": data["angle_1"],
            "ydata": data["wavelength_1"],
            "marker": "x",
            "x_error": [0.02],
            "label":"Zn Spektrum mit Winkeln von Unbekannter Lampe"
        },
        {
            "xdata": [48.27],
            "ydata": [636.6],
            "marker": "x",
            "x_error": [0.02],
            "label": "Zn 546.1nm mit Originalwinkel ",
            "color": "red"
        },
        #{
        #    "ydata":Zn_spectrum,
        #    "xdata": [find_x_for_y(y, spline, initial_guess) for y, initial_guess in zip(Zn_spectrum, initial_guesses)],
        #    "marker":"+",
        #    "label":"Zn Spektrum",
        #    "line":"None"
        #}
    ],

    x_label="relativer Ablenkwinkel γ/°", 
    y_label="Wellenlänge λ/nm",
    color_seed=78,
    plot=False
)

data["spline_wavelength"] = [round(wavelength, 1) for wavelength in spline(data["gamma_2"])]

#color_definitions = generate_latex_color_definitions(data["wavelength_2"])
#for color_def in color_definitions:
#    print(color_def)
data["colors_2"] = apply_color_to_text(data["colors_2"], data["wavelength_2"])


headers_2 = {
    "angle_2": {"label": "{Position(°)}", "precision": 2},
    "gamma_2": {"label": "{rel.Pos. $\\gamma$(°)}", "precision": 2},
    "colors_2": {"label": "{Farbe}"},
    "Intensity 2": {"label": "{Intensität}"},
    "spline_wavelength": {"label": "{$\\lambda_{Spline}$(nm)}"},
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



def sellmeier_eq(wavelength, B1, C1, B2, C2, B3, C3):
    term1 = (abs(B1) * wavelength**2) / (wavelength**2 - C1)
    term2 = (abs(B2) * wavelength**2) / (wavelength**2 - C2)
    term3 = (abs(B3) * wavelength**2) / (wavelength**2 - C3)
    return np.sqrt(1 + term1 + term2 + term3)

initial_guess = [1.34533359, 0.00997743871, 0.209073176, 0.0470450767, 0.937357162, 111.886764]
guess =  [0.955092741, -0.0196877842,  0.664587650,  0.04207532650, 29.7501402,  545.866345]

n = [sellmeier_eq(wavelength* 1e-3, *initial_guess) for wavelength in y]
n_spline = [sellmeier_eq(wavelength* 1e-3, *initial_guess) for wavelength in wavelength_smooth]

a, da, b, db, R2, s2 = main(x, n, [0.02] * len(x))

plot_data(
    filename="Plots/OPS_2.pdf",
    datasets=[
        {
            "xdata": x,
            "ydata": n,
            "x_error": [0.02],
            "label": "Brechungsindex n",
            "marker": ".",
        },
        {
            "xdata": x,
            "ydata": [a + b * x for x in x],
            "label": "Fit",
            "marker": "None",
        },
        {
            "xdata": angle_smooth,
            "ydata": n_spline,
            "label": "Spline Fit",
            "marker": "None",
        }
    ],
    x_label="relativer Ablenkwinkel γ/°",
    y_label="Brechungsindex n",
    color_seed=43,
    plot=False
)


data_val_2 = sorted(zip(data["gamma_3"], data["wavelength_3"]))
x3 = [point[0] for point in data_val_2]
y3 = [point[1] for point in data_val_2]

n3_0 = [np.sin(np.radians((angle + 60)/2)) / np.sin(np.radians(30)) for angle in x3]
y3, n3 = zip(*sorted(zip(y3, n3_0)))
y3 = np.array(y3) * 1e-3

dn_3 = [
    0.5 * np.cos(np.radians((angle + 60) / 2)) / np.sin(np.radians(30)) * (0.02 * np.pi / 180)
    for angle in x3
]

spline_3 = UnivariateSpline(y3, n3, s=500, k=4)
wavelength_smooth_3 = np.linspace(min(y3), max(y3), 500)
n3_spline = [spline_3(wavelength) for wavelength in wavelength_smooth_3]

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

initial_guess = [1.34533359, 0.00997743871, 0.209073176, 0.0470450767] # 0.937357162, 111.886764

def sellmeier_eq(wavelength, B1, C1, B2, C2):
    term1 = (abs(B1) * wavelength**2) / (wavelength**2 - C1)
    term2 = (abs(B2) * wavelength**2) / (wavelength**2 - C2)
    return np.sqrt(1 + term1 + term2 )


popt, pcov = curve_fit(sellmeier_eq, y3, n3, p0=initial_guess, sigma=dn_3, absolute_sigma=True)
print(np.linalg.cond(pcov))
print(np.diag(pcov))

print("B1, C1, B2, C2, B3, C3:", popt)

wavelengths_fit = np.linspace(y3.min(), y3.max(), 200)
n_fit = sellmeier_eq(wavelengths_fit, *popt)

(mean_spline, lower_bound, upper_bound, 
 mean_derivative, lower_bound_derivative, upper_bound_derivative,
 spline_predictions, spline_derivatives) = monte_carlo_spline_uncertainty(
    y3, n3, y_error=dn_3,
    spline_kwargs={'s':100, 'k':4}, 
    n_simulations=1000, x_smooth=wavelength_smooth_3, confidence=95)



plot_data(
    filename="Plots/OPS_4.pdf",
    datasets=[
        {
            "xdata": y3,
            "ydata": n3,
            "y_error": dn_3,
            "label": "n(λ)",
            "marker": ".",
            "line": "None",
        },
        {
            "xdata": wavelength_smooth_3,
            "ydata": n3_spline,
            "label": "Spline Fit",
            "marker": "None",
        },
        {
            "confidence": True,
            "xdata": wavelength_smooth_3,
            "low_bound": lower_bound,
            "high_bound": upper_bound,
            "label": "95% Konfidenzintervall",
        },
        {
            "xdata": wavelengths_fit,
            "ydata": n_fit,
            "label": "Sellmeier Fit",
            "marker": "None",
        },
        #{
        #    "xdata": wavelengths_fit,
        #    "ydata": refractive_index_polynomial(wavelengths_fit),
        #    "label": "Sellmeier Fit 2",
        #    "marker": "None",
        #}

    ],
    x_label="Wellenlänge λ/µm",
    y_label="Brechungsindex n",
    color_seed=43,
    plot=False
)

def sellmeier_derivative(wavelength, B1, C1, B2, C2):
    n = sellmeier_eq(wavelength, B1, C1, B2, C2)
    term1 = (B1 * wavelength * C1) / (wavelength**2 - C1)**2
    term2 = (B2 * wavelength * C2) / (wavelength**2 - C2)**2
    dn_dlambda = -1 / n * (term1 + term2 )
    return dn_dlambda

data["n"], data["dn"], data["n_pow"] = map(list, zip(*(round_val(n, dn) for n, dn in zip(n3[::-1], dn_3[::-1]))))

data["wavelength_3"] = apply_color_to_text(data["wavelength_3"], data["wavelength_3"])

headers_3 = {
    "angle_3": {"label": "{Position(°)}", "precision": 2},      
    "gamma_3": {"label": "{rel.Pos. $\\gamma$(°)}", "precision": 2},
    "n": {"label": "{n $\\pm \\Delta n$}", "precision": data["n_pow"], "err": data["dn"]},                 
    "wavelength_3": {"label": "{$\\lambda$(nm)}"}
}

print_standard_table(
    data=data,
    headers=headers_3,
    column_formats= ["3.2", "3.2", "1.4", "1.4", "2.1"],
    caption="Messaufnahme des Hg Spektrum wobei alle Winkeln die minimale Ablenkung für jede Wellenlänge im Prisma darstellen.",
    label="tab:HgMinima",
    show=False
)

spline_derivative = spline_3.derivative()

dn_dlambda_spline_577 = spline_derivative(577 * 1e-3)
dn_dlambda_spline_579 = spline_derivative(579.1 * 1e-3)
dn_dlambda_577 = sellmeier_derivative(577*1e-3, *popt)
dn_dlambda_579 = sellmeier_derivative(579*1e-3, *popt)

plot_data(
    filename="Plots/OPS_5.pdf",
    datasets=[
        {
            "xdata":wavelength_smooth_3,
            "ydata":spline_derivative(wavelength_smooth_3),
            "label":"Ableitung Spline Fit",
            "marker":"None",
        },
        {
            "confidence": True,
            "xdata": wavelength_smooth_3,
            "low_bound": lower_bound_derivative,
            "high_bound": upper_bound_derivative,
            "label": "95% Konfidenzintervall",
        },
        {
            "xdata":wavelength_smooth_3,
            "ydata":[sellmeier_derivative(wavelength, *popt) for wavelength in wavelength_smooth_3],
            "label":"Ableitung Sellmeier Fit",
            "marker":"None",
        },
        {
            "ydata": [dn_dlambda_577, dn_dlambda_579],
            "xdata": [0.577, 0.5791],
            "line":"None",
            "marker":"x",
            "label":"Ableitung Sellmeier Fit an 577nm und 579nm"
        },
        {
            "ydata": [dn_dlambda_spline_577, dn_dlambda_spline_579],
            "xdata": [0.577, 0.5791],
            "marker":"x",
            "line":"None",
            "label":"Ableitung Spline Fit an 577nm und 579nm"
        }
    ],
    x_label="Wellenlänge λ/µm",
    y_label="dn/dλ",
    color_seed=43,
    plot=False
)




print(f"dn/dλ at 577 nm: {round_val( dn_dlambda_577, intermed=True)[0]} | dn/dλ at 577 nm spline: {round_val( float(dn_dlambda_spline_577), intermed=True)[0]}" )
print(f"dn/dλ at 579 nm: {round_val( dn_dlambda_579, intermed=True)[0]} | dn/dλ at 579 nm spline: {round_val( float(dn_dlambda_spline_579), intermed=True)[0]}" )


resolve_theo = 0.5 * (577 + 579.1) / (579.1 - 577)
resolve_exp_1 =  2.04 * 1e-3 / np.cos(np.radians( (data["gamma_3"][1]+60)/2) ) * abs(dn_dlambda_577) *1e6
resolve_exp_2 =  2.04 * 1e-3 / np.cos(np.radians( (data["gamma_3"][0]+60)/2) ) * abs(dn_dlambda_579) *1e6
print(f"{round_val(resolve_theo, intermed=False)[0]}, {round_val(resolve_exp_1, intermed=False)[0]},  {round_val(resolve_exp_2, intermed=False)[0]}")

resolve_exp_1 =  2.04 * 1e-3 / np.cos(np.radians( (data["gamma_3"][1]+60)/2) ) * abs(dn_dlambda_spline_577) *1e6
resolve_exp_2 =  2.04 * 1e-3 / np.cos(np.radians( (data["gamma_3"][0]+60)/2) ) * abs(dn_dlambda_spline_579) *1e6

print(resolve_exp_1, resolve_exp_2)

