from calendar import c
from wsgiref import headers
import numpy as np
from decimal import Decimal
from scipy import constants

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from Functions.help import *    # lmfit, last_digit, print_round_val
from Functions.plotting import *
from Functions.tables import *


factor = 2 * np.sqrt(2 * np.log(2))

cos2 = lambda x, a, c, d: a * np.cos(np.radians(x - c))**2 + d

def calc_Intensity_error(I):
    I_dec = Decimal(str(I))
    ld = last_digit(I_dec)
    return float(ld*2 + 0.01*float(I_dec))

# Task 2

data2 = {
    'Angle': np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 100, 110, 120, 130, 140, 150, 160, 170, 175, 180, 190, 200]),
    'dAngle': 1,
    'Intensity': ["0.00745", "0.118", "0.358", "0.700", "1.10", "1.52", "1.90", "2.13", "2.33", "2.36", "2.33", "2.22", "1.97", "1.63", "1.24", "0.835", "0.465", "0.183", "0.0307", 0.00215, 0.0168, 0.127, 0.375],
    'dIntensity': [],
    'Baseline': [4.02, 4.04, 4.04, 4.05, 4.06, 4.06, 4.06, 4.07, 4.08, 4.09, 4.08, 4.09, 4.09, 4.09, 4.09, 4.09, "4.10", "4.10", "4.10", 4.11, "4.10", 4.11, 4.11]
}

angles = data2['Angle']
Intensity = np.array([float(Decimal(str(i))) for i in data2['Intensity']])
dIntensity = np.array([calc_Intensity_error(i) for i in data2['Intensity']])
Baseline = np.array([float(Decimal(str(b))) for b in data2['Baseline']])

Intensity = Intensity * max(Baseline)/Baseline

data2.update({'Intensity': Intensity, 'dIntensity': dIntensity, 'Baseline': Baseline, 'dAngle': data2['dAngle']*np.ones_like(angles)})



init = {'a':5, 'c':90, 'd':0.01}
results = lmfit(data2['Angle'], data2['Intensity'], model=cos2, initial_params=init, yerr=data2['dIntensity'])

fitx = np.linspace(0, 200, 500)

s2 = DatasetSpec(x=data2['Angle'], y=data2['Intensity'],
                yerr=data2['dIntensity'], 
                xerr=data2['dAngle'],
                fit_y=results.eval(x=fitx), fit_x=fitx,
                label="Measured Data")

plot_data(
    s2,
    title="Laser Intensity vs Angle",
    xlabel="Angle (°)",
    ylabel="Intensity (mW)",
    filename=repo_root / "FP" / "Laser" / "A2.pdf",
    plot=False
)

headers2 = {
    'Angle':     {'label': '{Winkel (°)}', 'intermed': False, 'err': data2['dAngle']},
    'Intensity': {'label': '{Intensität (mW)}', 'intermed': False, 'err': data2['dIntensity']},
}

print_standard_table(
    data=data2,
    headers=headers2,
    caption="Gemessene Winkelabhängigkeit von Justierlaserintensität durch Polarisationsfilter.",
    label="tab:A2",
    show=True
)


L = 4.13e-3
T1 = 6.5e-6
dT1 = 0.05e-6
T2 = 2.26e-6
dT2 = 0.05e-6

print(f"T1 Mirror Transmissivity:{T1/L*100:.2g} % ± {dT1/L*100:.2g} %")
print(f"T2 Mirror Transmissivity:{T2/L*100:.2g} % ± {dT2/L*100:.2g} %")
print('-'*40)

# Task 6

data6 = {
    'Angle': np.array([0, 10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100, 110, 120, 130, 140, 150, 160, 165, 170, 180, 190, 200]),
    'dAngle': 1,
    'Intensity': ["100.0", 90.7, 75.2, 56.9, 39.8, 23.8, 11.1, 3.28, 1.39, 1.07, 5.08, 14.7, "29.0", 46.3, 63.4, 77.2, 93.7, "101.0", "101.0", "103.0", "98.0", 90.6, 72.7],
    'dIntensity': [],
    'Baseline': np.array([182, 183, 180, 178, 180, 178, 178, 178, 178, 179, 178, 178, 178, 178, 178, 179, 180, 179, 179, 178, 178, 179, 178]),
}



angles6 = data6['Angle']
Intensity6 = np.array([float(Decimal(str(i))) for i in data6['Intensity']])/1e3
dIntensity6 = np.array([calc_Intensity_error(i) for i in data6['Intensity']])/1e3
Baseline6 = data6['Baseline']/1e3

Intensity6 = Intensity6 *max(Baseline6)/Baseline6

data6.update({'Intensity': Intensity6, 'dIntensity': dIntensity6, 'Baseline': Baseline6, 'dAngle': data6['dAngle']*np.ones_like(data6['Angle'])})

initcos6 = {'a':0.1, 'c':170, 'd':0.001}
results6 = lmfit(data6['Angle'], data6['Intensity'], model=cos2, initial_params=initcos6, yerr=data6['dIntensity'])


s6 = DatasetSpec(x=data6['Angle'], y=data6['Intensity'],
                yerr=data6['dIntensity'], xerr=data6['dAngle'],
                label="Measured Data",
                fit_y=results6.eval(x=fitx), fit_x=fitx,    
)

plot_data(
    s6,
    title="Laser Intensity vs Angle (High Power)",
    xlabel="Angle (°)",
    ylabel="Intensity (mW)",
    filename=repo_root / "FP" / "Laser" / "A6.pdf",
    plot=False
)

headers6 = {
    'Angle':     {'label': '{Winkel (°)}', 'intermed': False, 'err': data6['dAngle']},
    'Intensity': {'label': '{Intensität (mW)}', 'intermed': False, 'err': data6['dIntensity']},
}

print_standard_table(
    data=data6,
    headers=headers6,
    caption="Gemessene Winkelabhängigkeit von Messlaserintensität durch Polarisationsfilter.",
    label="tab:A6",
    show=True
)



print('-'*40)
print('Fit Results for Task 2 and 6:')
print(f"$c_{{Just}} = {print_round_val(results.params['c'].value, results.params['c'].stderr)} $°")

print(f"$c_{{Mess}} = {print_round_val(results6.params['c'].value, results6.params['c'].stderr)} $°")


"""
print(results.fit_report())
print('-'*40)
print(results6.fit_report())
print('-'*40)"""

print('-'*40)


# Task 3
t_FP = 2.740e-3     # difference Peak 2 to 5 in seconds
t_mode = 600e-6    # difference Peak 5 to 6
FWHM = 60e-6     # FWHM Peak 2,3 and 5,6

t_err = FWHM /(2 * np.sqrt(2 * np.log(2))) * np.sqrt(2)   # error propagation for time measurements for both together

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
I_ND = 1.55e-3  # no discharge intensity
I_WD = 1.59e-3  # with discharge intensity

print(f" normalized difference {print_round_val((I_WD - I_ND) / (I))*100}%")
print(f" gain factor {print_round_val(((I_WD)/(I_ND))*100)}%")
print('-'*40)

#Task 7

S2 = (26.5 + 32.45)/2

data7 = {
    'Position': np.array([76, 76.5, 77, 77.5, 78, 78.5, 79, 79.5, 80, 80.5, 81, 81.5, 82, 82.5, 82, 81.5, 81, 80.5, 80, 79, 76.5]),
    'dPosition': 0.1*np.sqrt(2)+0.1,
    'Intensity': [89.3, 105, 91.5, 69.5, 73.3, "59.0", "53.0","20.0", 2.70, "3.00", "2.10", 2.23, 2.13, 2.15, 2.14, 2.12, 2.07, "2.10", 11.2, 13.5, 66.6],
    'dIntensity': np.array([0.3, 0.3, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.3, 0.2, 0.3]),
}
data7['Position'] = data7['Position'] - S2

mask = slice(1, -2)
mask_rest = np.ones_like(data7['Position'], dtype=bool)
mask_rest[mask] = False


data7.update({'dPosition': np.array(data7['dPosition'] * np.ones_like(data7['Position'])), 'Intensity': np.array([float(Decimal(str(i))) for i in data7['Intensity']]), 'dIntensity': np.array([calc_Intensity_error(i) for i in data7['Intensity']])})

hinge = lambda x, m, L0, Pbg: Pbg + np.maximum(0.0, m*(L0 - x))

results7 = lmfit(data7['Position'][mask], data7['Intensity'][mask], model=hinge, initial_params={'m':23, 'L0':51, 'Pbg':2.2})

fit_pos = results7.eval(x=data7['Position'][mask])

sigma = np.sqrt(1/(len(data7['Position'][mask]) - len(results7.params)) * np.sum(((data7['Intensity'][mask] - fit_pos))**2 ) )

print(f"sigma: {sigma}")

fitx = np.linspace(data7['Position'].min(), data7['Position'].max(), 400)





print(results7.fit_report())
print('-'*40)

print(f"Radius of sperical resonator: R = {print_round_val(results7.params['L0'].value, results7.params['L0'].stderr)} cm")
print('-'*40)

s7 = DatasetSpec(x=data7['Position'][mask], y=data7['Intensity'][mask], marker='x',
                yerr=data7['dIntensity'][mask], xerr=data7['dPosition'][mask],
                fit_y=results7.eval(x=fitx), fit_x=fitx,
                label="Measured Data", confidence=calc_CI(results7, fitx, sigmas=(1,)),
)

s7_rest = DatasetSpec(x=data7['Position'][mask_rest], y=data7['Intensity'][mask_rest], marker='x',
                yerr=data7['dIntensity'][mask_rest], xerr=data7['dPosition'][mask_rest],
                label="Outliers")

headers7 = {
    'Position':     {'label': '{Länge L (cm)}', 'intermed': False, 'err': data7['dPosition']},
    'Intensity': {'label': '{Intensität $(\\mu W)$}', 'intermed': False, 'err': data7['dIntensity']},
}

plot_data(
    [s7, s7_rest],
    title="Cavity Length vs Light Intensity",
    xlabel="Länge L (cm)",
    ylabel="Intensität (μW)",
    filename=repo_root / "FP" / "Laser" / "A7.pdf",
    plot=False
)

print_standard_table(
    data=data7,
    headers=headers7,
    caption="Gemessene Abhängigkeit der Lichtintensität von der Kavitätslänge.",
    label="tab:A7",
    show=True)


# Task 8+9

data9 = {
    'Position': np.array([76.5, 77.5, 78.5, 79.5]),
    'dPosition': [0.4],
    'delta': np.array([468e-6, 460e-6, 436e-6, 440e-6]),
    'fwhm1': np.array([32e-6, 36e-6, 36e-6, 40e-6]),
    'fwhm2': np.array([36e-6, 36e-6, 40e-6, 40e-6])
}

data9['Position'] = (data9['Position'] - S2)*1e-2  # convert to meters
data9['dPosition'] = np.array(data9['dPosition'] * np.ones_like(data9['Position'])) * 1e-2  # convert to meters



delta_err = np.sqrt((data9['fwhm1'])**2 + (data9['fwhm2'])**2)/factor

print(delta_err)

freq = v_FSR_FP * data9['delta'] / t_FP
freq_err = freq * np.sqrt( (delta_err/data9['delta'])**2 + (v_FSR_mode_err/v_FSR_mode)**2 + (t_err/t_FP)**2 )

fitx = np.linspace(data9['Position'].min(), data9['Position'].max(), 50)

results9 = lmfit(1/data9['Position'], freq, yerr=freq_err, model=lambda x, a: constants.c/2 * a * x,
               initial_params={'a':1})

s9 = DatasetSpec(x=1/data9['Position'], y=freq,
                yerr=freq_err, xerr=(data9['dPosition']/data9['Position']**2),
                label="Gemessene Daten", fit_y=results9.eval(x=1/fitx), fit_x=1/fitx,
                confidence=calc_CI(results9, 1/fitx, sigmas=(1,))
)

headers9 = {
    'Position':     {'label': '{Position (cm)}', 'intermed': False, 'err': data9['dPosition']*100, 'data': data9['Position']*100},
    'delta': {'label': '{Zeitdifferenz $\\Delta t (\\mu s)$}', 'intermed': True, 'err': delta_err*1e6, 'data': data9['delta']*1e6},
    'Frequency': {'label': '{FSR (MHz)}', 'intermed': True, 'err': freq_err*1e-6, 'data': freq*1e-6},
}
s10 = DatasetSpec(x=1/fitx, y=constants.c/2 * (1/fitx), label="Theoretische Vorhersage", line='--', marker='None')

print(results9.fit_report())
print('-'*40)
print(f"a = ${print_round_val(results9.params['a'].value, results9.params['a'].stderr)}$")
print('-'*40)

plot_data(
    [s9, s10],
    title="Frequenzabweichung vs Position",
    xlabel="1/Position (1/m)",
    ylabel="Frequenzabweichung (Hz)",
    filename=repo_root / "FP" / "Laser" / "A9.pdf",
    plot=False
)

print_standard_table(
    data=data9,
    headers=headers9,
    caption="Gemessene Frequenzabweichung in Abhängigkeit der Position der Spiegel.",
    label="tab:A9",
    show=True
)
