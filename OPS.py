from calendar import c
import numpy as np
from plotting_minus import plot_data
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import math
from help import *
from scipy.optimize import curve_fit

data = {
    "angle": [131.69, 132.03, 132, 132.36, 132.64, 132.74, 132.82, 129.15, 129.25, 130.03, 131.1, 131.17],
    "wavelength": [546.1, 579.1, 577, 623.4, 671.6, 690.7, 708.2, 404.7, 407.8, 435.8, 491.6, 496],
    "angle_2":[132.43, 130.93, 130.77, 130.7, 131.35, 131.47, 132.2, 132.31, 130.59 ],
    "angle_3":[132.2, 132.18, 131.86, 131.17, 130.09, 129.22],
    "wavelength_3":[579.1, 577, 546.1, 491.6, 435.8, 404.7]	
}

data["angle"] = [180.8 - angle for angle in data["angle"]]
data["angle_2"] = [180.8 - angle for angle in data["angle_2"]]
data["angle_3"] = [180.8 - angle for angle in data["angle_3"]]

data_val = sorted(zip(data["angle"], data["wavelength"]))
x = [point[0] for point in data_val]
y = [point[1] for point in data_val]

x2 = sorted(data["angle_2"])


spline = UnivariateSpline(x, y, s=500, k=4) 
# Generate points for plotting the spline fit
angle_smooth = np.linspace(min(x), max(x), 200)
wavelength_smooth = spline(angle_smooth)


"""
plot_data(
    filename="Plots/OPS_1.pdf",
    datasets=[
        {
            "xdata": x,
            "ydata": y,
            "x_error": [0.01] * len(x),
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
            "xdata": x2,
            "ydata": spline(x2),
            "marker": "*",
            "x_error": [0.01] * len(x2),
            "label":"Unbekannte Lampe mit λ von Kalibrierkurve"
        },
        {
            "xdata": [47.57, 49.07, 49.23, 49.3, 48.65, 47.8],
            "ydata": [636.6, 481.1, 472.2, 468, 518.2, 602.1 ],
            "marker": "x",
            "x_error": [0.01] * 6,
            "label":"Zn Spektrum mit Winkeln von Unbekannter Lampe"
        },
        {
            "xdata": [47.47],
            "ydata": [636.6],
            "marker": "x",
            "x_error": [0.01],
            "label": "Zn 546.1nm mit Originalwinkel ",
            "color": "red"
        },
    ],

    x_label="Angle/°", 
    y_label="Wavelength λ/nm",
    color_seed=78,
)
"""
def calculate_refractive_index(wavelength):
    wavelength = wavelength * 1e-3 
    
    term1 = (1.59034337 * wavelength**2) / (wavelength**2 - 0.0093273434)
    term2 = (0.138464579 * wavelength**2) / (wavelength**2 - 0.0427498654)
    term3 = (1.21988043 * wavelength**2) / (wavelength**2 - 119.251777)
    
    n2 = term1 + term2 + term3 + 1
    return np.sqrt(n2)

n = [calculate_refractive_index(wavelength) for wavelength in y]
n_spline = [calculate_refractive_index(wavelength) for wavelength in wavelength_smooth]

a,da, b, db, R2, s2 = main(x, n, [0.01] * len(x))
"""
plot_data(
    filename="Plots/OPS_2.pdf",
    datasets=[
        {
            "xdata": x,
            "ydata": n,
            "x_error": [0.01] * len(y),
            "label": "Refractive Index",
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
)
"""

data_val_2 = sorted(zip(data["angle_3"], data["wavelength_3"]))
x3 = [point[0] for point in data_val]
y3 = [point[1] for point in data_val]

spline_3 = UnivariateSpline(x3, y3, s=50, k=4)
angle_smooth_3 = np.linspace(min(x3), max(x3), 500)
wavelength_smooth_3 = spline_3(angle_smooth_3)


"""
plot_data(
    filename="Plots/OPS_3.pdf",
    datasets=[
        {
            "xdata": x3,
            "ydata": y3,
            "x_error": [0.01] * len(x),
            "label": "Hg Spektrum",
            "marker": ".",
        },
        {
            "xdata": angle_smooth_3,
            "ydata": wavelength_smooth_3,
            "marker": "None",
            "label": "Spline Fit Kalibrierkurve durch Hg Spektrum",
        },
    ],
    x_label="Angle/°", 
    y_label="Wavelength λ/nm",
    color_seed=78,
)
"""


n3 = [np.sin(np.radians((angle + 60)/2)) / np.sin(np.radians(30)) for angle in x3]
y3, n3 = zip(*sorted(zip(y3, n3)))
y3 = np.array(y3) * 1e-3

spline_3 = UnivariateSpline(y3, n3, s=500, k=4)
wavelength_smooth_3 = np.linspace(min(y3), max(y3), 500)
n3_spline = [spline_3(wavelength) for wavelength in wavelength_smooth_3]

def sellmeier_eq(wavelength, B1, C1, B2, C2, B3, C3):
    term1 = (B1 * wavelength**2) / (wavelength**2 - C1)
    term2 = (B2 * wavelength**2) / (wavelength**2 - C2)
    term3 = (B3 * wavelength**2) / (wavelength**2 - C3)
    return np.sqrt(1 + term1 + term2 + term3)

initial_guess = [1.1515019, 0.010598413, 0.118583612, 0.011822519, 1.26301359, 129.617662]
popt, pcov = curve_fit(sellmeier_eq, y3, n3, p0=initial_guess)

print("B1, C1, B2, C2, B3, C3:", popt)

wavelengths_fit = np.linspace(y3.min(), y3.max(), 200)
n_fit = sellmeier_eq(wavelengths_fit, *popt)

plot_data(
    filename="Plots/OPS_4.pdf",
    datasets=[
        {
            "xdata": y3,
            "ydata": n3,
            "label": "n(λ)",
            "marker": "o",
            "line": "None",
        },
        {
            "xdata": wavelength_smooth_3,
            "ydata": n3_spline,
            "label": "Spline Fit",
            "marker": "None",
        },
        {
            "xdata": wavelengths_fit,
            "ydata": n_fit,
            "label": "Sellmeier Fit",
            "marker": "None",
        }
    ],
    x_label="Wavelength λ/µm",
    y_label="Brechungsindex n",
    color_seed=43,
)

def sellmeier_derivative(wavelength, B1, C1, B2, C2, B3, C3):
    n = sellmeier_eq(wavelength, B1, C1, B2, C2, B3, C3)
    term1 = (B1 * wavelength * C1) / (wavelength**2 - C1)**2
    term2 = (B2 * wavelength * C2) / (wavelength**2 - C2)**2
    term3 = (B3 * wavelength * C3) / (wavelength**2 - C3)**2
    dn_dlambda = -1 / n * (term1 + term2 + term3)
    return dn_dlambda

dn_dlambda_577 = sellmeier_derivative(577*1e-3, *popt)
dn_dlambda_579 = sellmeier_derivative(579*1e-3, *popt)


print("dn/dλ at 577 nm:",  dn_dlambda_577)
print("dn/dλ at 579 nm:",  dn_dlambda_579)


resolve_theo = 0.5 * (577 + 579.1) / (579.1 - 577)
resolve_exp_1 =  2.04 * 1e-3 / np.cos(np.radians( (data["angle_3"][1]+60)/2) ) * abs(dn_dlambda_577) *1e6
resolve_exp_2 =  2.04 * 1e-3 / np.cos(np.radians( (data["angle_3"][0]+60)/2) ) * abs(dn_dlambda_579) *1e6

print(resolve_theo, resolve_exp_1, resolve_exp_2)

