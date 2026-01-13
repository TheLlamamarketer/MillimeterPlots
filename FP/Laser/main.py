import numpy as np
from decimal import Decimal
from scipy import constants

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from Functions.help import *
from Functions.plotting import *



data2 = {
    'Angles': np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 100, 110, 120, 130, 140, 150, 160, 170, 175, 180, 190, 200]),
    'dAngles': [1],
    'Intensity': ["0.00745", "0.118", "0.358", "0.700", "1.10", "1.52", "1.90", "2.13", "2.33", "2.36", "2.33", "2.22", "1.97", "1.63", "1.24", "0.835", "0.465", "0.183", "0.0307", 0.00215, 0.0168, 0.127, 0.375],
    'dIntensity': [],
    'Baseline': [4.02, 4.04, 4.04, 4.05, 4.06, 4.06, 4.06, 4.07, 4.08, 4.09, 4.08, 4.09, 4.09, 4.09, 4.09, 4.09, "4.10", "4.10", "4.10", 4.11, "4.10", 4.11, 4.11]
}

angles = data2['Angles']
Intensity = np.array([float(Decimal(i)) for i in data2['Intensity']])
dIntensity = np.array([last_digit(Decimal(i))*1 for i in data2['Intensity']])
Baseline = np.array([float(Decimal(b)) for b in data2['Baseline']])

Intensity = Intensity * max(Baseline)/Baseline



cos2 = lambda x, a, c, d: a * np.cos(np.radians(x - c))**2 + d


init = {'a':5, 'c':90, 'd':0.01}
results = lmfit(angles, Intensity, model=cos2, initial_params=init, yerr=None)

fitx = np.linspace(0, 200, 500)

s = DatasetSpec(x=angles, y=Intensity,
                yerr=dIntensity, 
                xerr=data2['dAngles']*np.ones_like(angles),
                fit_y=results.eval(x=fitx), fit_x=fitx,
                label="Measured Data")

plot_data(
    s,
    title="Laser Intensity vs Angle",
    xlabel="Angle (°)",
    ylabel="Intensity (mW)",
    filename=repo_root / "FP" / "Laser" / "A2.pdf",
    plot=False
)

# Task 6

data6 = {
    'Angles': np.array([0, 10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100, 110, 120, 130, 140, 150, 160, 165, 170, 180, 190, 200]),
    'dAngles': [1e-3],
    'Intensity': ["100.0", 90.7, 75.2, 56.9, 39.8, 23.8, 11.1, 3.28, 1.39, 1.07, 5.08, 14.7, "29.0", 46.3, 63.4, 77.2, 93.7, "101.0", "101.0", "103.0", "98.0", 90.6, 72.7],
    'dIntensity': [],
    'Baseline': np.array([182, 183, 180, 178, 180, 178, 178, 178, 178, 179, 178, 178, 178, 178, 178, 179, 180, 179, 179, 178, 178, 179, 178]),
}

angles2 = data6['Angles']
Intensity2 = np.array([float(Decimal(i)) for i in data6['Intensity']])/1e3
dIntensity2 = np.array([last_digit(Decimal(i))*1 for i in data6['Intensity']])/1e3
Baseline2 = data6['Baseline']/1e3

Intensity2 = Intensity2 *max(Baseline2)/Baseline2

initcos2 = {'a':0.1, 'c':170, 'd':0.001}
results2 = lmfit(angles2, Intensity2, model=cos2, initial_params=initcos2, yerr=dIntensity2)


s2 = DatasetSpec(x=angles2, y=Intensity2,
                yerr=dIntensity2, xerr=data6['dAngles']*np.ones_like(angles2),
                label="Measured Data",
                fit_y=results2.eval(x=fitx), fit_x=fitx,
)

plot_data(
    s2,
    title="Laser Intensity vs Angle (High Power)",
    xlabel="Angle (°)",
    ylabel="Intensity (mW)",
    filename=repo_root / "FP" / "Laser" / "A6.pdf",
    plot=False
)

print('-'*40)
print(results.fit_report())
print('-'*40)
print(results2.fit_report())
print('-'*40)


# Task 3
t_FP = 2.740e-3     # difference Peak 2 to 5
t_mode = 600e-6    # difference Peak 5 to 6
P2_FWHM = 60e-6     # FWHM Peak 2 and 5
P3_FWHM = 60e-6     # FWHM Peak 3 and 6

t_err = P2_FWHM /(2 * np.sqrt(2 * np.log(2)))

v_FSR_FP = 2e9

v_FSR_mode = v_FSR_FP * t_mode / t_FP

v_FSR_mode_err = v_FSR_mode * t_err * np.sqrt( (1/t_mode)**2 + (1/t_FP)**2 )

L = constants.c / (2 * v_FSR_mode)
L_err = L * v_FSR_mode_err / v_FSR_mode

print(f"Cavity Length: L= {print_round_val(L*1e2, L_err*1e2)} cm")
print(f"Free Spectral Range of Mode Spacing: \\nu_{{FSR}} = {print_round_val(v_FSR_mode*1e-6, v_FSR_mode_err*1e-6)} MHz")
print('-'*40)


# Task 5
I = 3.33e-3     # original laser intensity
I_off = 773e-6
I_ND = 1.55e-3  # no discharge intensity
I_WD = 1.59e-3  # with discharge intensity
 
amplification = (I_WD - I_ND) / I
print(f"Photodiode Amplification Factor: {print_round_val(amplification)*100}%")

#Task 7
S2 = (26.5 + 32.45)/2

data7 = {
    'Position': np.array([76, 76.5, 77, 77.5, 78, 78.5, 79, 79.5, 80, 80.5, 81, 81.5, 82, 82.5, 82, 81.5, 81, 80.5, 80, 79, 76.5]),
    'dPosition': [0.1],
    'Intensity': np.array([89.3, 105, 91.5, 69.5, 73.3, 59.0, 53.0, 20.0, 2.70, 3.00, 2.10, 2.23, 2.13, 2.15, 2.14, 2.12, 2.07, 2.10, 11.2, 13.5, 66.6]),
    'dIntensity': np.array([0.3, 0.3, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.3, 0.2, 0.3]),
}
data7['Position'] = data7['Position'] - S2

mask = slice(1, -2)
mask_rest = np.ones_like(data7['Position'], dtype=bool)
mask_rest[mask] = False

data7['dIntensity'] = np.sqrt(data7['dIntensity']**2)

hinge = lambda x, m, L0, Pbg: Pbg + np.maximum(0.0, m*(L0 - x))

results7 = lmfit(data7['Position'][mask], data7['Intensity'][mask], yerr=None, model=hinge, initial_params={'m':23, 'L0':51, 'Pbg':2.2})

fitx = np.linspace(data7['Position'].min(), data7['Position'].max(), 400)

print(results7.fit_report())
print('-'*40)

print(f"Radius of sperical resonator: R = {print_round_val(results7.params['L0'].value, results7.params['L0'].stderr)} cm")
print('-'*40)

s7 = DatasetSpec(x=data7['Position'][mask], y=data7['Intensity'][mask], marker='x',
                yerr=data7['dIntensity'][mask], xerr=data7['dPosition']*np.ones_like(data7['Position'][mask]),
                fit_y=results7.eval(x=fitx), fit_x=fitx,
                label="Measured Data")

s7_rest = DatasetSpec(x=data7['Position'][mask_rest], y=data7['Intensity'][mask_rest], marker='x',
                yerr=data7['dIntensity'][mask_rest], xerr=data7['dPosition']*np.ones_like(data7['Position'][mask_rest]),
                label="Outliers")


plot_data(
    [s7, s7_rest],
    title="Cavity Length vs Light Intensity",
    xlabel="Position (cm)",
    ylabel="Intensity (μW)",
    filename=repo_root / "FP" / "Laser" / "A7.pdf",
    plot=False
)


# Task 9

data9 = {
    'Position': np.array([76.5, 77.5, 78.5, 79.5]),
    'dPosition': [0.4],
    'delta': np.array([468e-6, 460e-6, 436e-6, 440e-6]),
    'fwhm1': np.array([32e-6, 36e-6, 36e-6, 40e-6]),
    'fwhm2': np.array([36e-6, 36e-6, 40e-6, 40e-6]),
}
data9['Position'] = data9['Position'] - S2
data9['Position'] = data9['Position'] * 1e-2  # convert to meters
data9['dPosition'] = np.array(data9['dPosition'] * np.ones_like(data9['Position'])) * 1e-2  # convert to meters

factor = 2 * np.sqrt(2 * np.log(2))
delta_err = np.sqrt((data9['fwhm1'])**2 + (data9['fwhm2'])**2)/factor

freq = v_FSR_FP * data9['delta'] / t_FP
freq_err = freq * np.sqrt( (delta_err/data9['delta'])**2 + (v_FSR_mode_err/v_FSR_mode)**2 + (t_err/t_FP)**2 )

fitx = np.linspace(data9['Position'].min(), data9['Position'].max(), 20)

results9 = lmfit(1/data9['Position'], freq, yerr=freq_err, model=lambda x, a: constants.c/2 * a * x,
               initial_params={'a':1})

s9 = DatasetSpec(x=1/data9['Position'], y=freq,
                yerr=freq_err, xerr=(data9['dPosition']/data9['Position']**2),
                label="Measured Data", fit_y=results9.eval(x=1/fitx), fit_x=1/fitx)

print(results9.fit_report())
print('-'*40)
plot_data(
    s9,
    title="Frequency Deviation vs Position",
    xlabel="Position (1/m)",
    ylabel="Frequency Deviation (Hz)",
    filename=repo_root / "FP" / "Laser" / "A9.pdf",
    plot=False
)
