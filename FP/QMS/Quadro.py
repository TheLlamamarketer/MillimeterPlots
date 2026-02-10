from re import S
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, peak_widths, peak_prominences
from matplotlib.ticker import MultipleLocator

from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).parent.parent.parent))

from Functions.plotting import *
from Functions.tables import *
from Functions.help import *
from scipy.interpolate import UnivariateSpline
from collections import defaultdict

def fwhm(m, y, peak_idx):
    """
    FWHM relative to baseline 0.
    Returns (dm, m_center, half_level). Returns (nan, nan, half) if not measurable.
    """
    m = np.asarray(m, float)
    y = np.asarray(y, float)

    ypk = y[peak_idx]
    if ypk <= 0:
        return np.nan, np.nan, np.nan

    half = 0.5 * ypk

    # left crossing
    i = peak_idx
    while i > 0 and y[i] > half:
        i -= 1
    mL = np.interp(half, [y[i], y[i+1]], [m[i], m[i+1]])

    # right crossing
    i = peak_idx
    while i < len(y) - 1 and y[i] > half:
        i += 1
    mR = np.interp(half, [y[i-1], y[i]], [m[i-1], m[i]])


    return mR, mL, mR - mL, half

source_dir = Path(__file__).resolve().parent / "Data"

massrs, signals = [], []
names, pressures = [], []
data = []

for f in source_dir.glob('*.txt'):
    p = f.stem.find('_p1')
    redo = f.stem.find('_redo')
    name = f.stem[:p] + f.stem[p+5:]
        # Change file if specific name matches
    if name == 'Air_1':
        f = Path(f'{source_dir}/Air_p1_2_U90_redo.txt')
        p -= 2
    
    if 'redo' in name:
        name = name.replace('_redo', '') 
    names.append(name)
    

    pressure = float(f.stem[p+2:p+5])/10
    
    s, mr = pd.read_csv(source_dir / f.name, sep='\t', header=None, decimal=',').values.T
    sort_idx = np.argsort(mr)
    mr, s = mr[sort_idx], s[sort_idx]
    data.append((mr*10, s/pressure))
    
    pressures.append(pressure)



runs = defaultdict(lambda: {"name": [], "data": [], "pressure": []})

for name, d, pressure in zip(names, data, pressures):
    if 'Air_U' in name:
        category = 'voltage_run'
    elif 'Air_s' in name:
        category = 'resolution_run'
    else:
        category = 'molecules_run'
    
    runs[category]["name"].append(name)
    runs[category]["data"].append(d)
    runs[category]["pressure"].append(pressure)

voltage_run = runs['voltage_run']
resolution_run = runs['resolution_run']
molecules_run = runs['molecules_run']

sorting = np.argsort([int(name[5:]) for name in voltage_run['name']])[::-1]
voltage_run['name'] = [voltage_run['name'][i] for i in sorting]
voltage_run['data'] = [voltage_run['data'][i] for i in sorting]
voltage_run['pressure'] = [voltage_run['pressure'][i] for i in sorting]

U_FR = 113.4
U_FA = np.array([110.8, 90.10, 69.95, 50.003, 29.973, 9.973]) 


U_B = U_FR - U_FA
l = 0.1
nu = 2.5e6


sU, s_peaks, cycles, deltas, masses = [], [], [], [], []
for i, (name, d, pressure) in enumerate(zip(voltage_run['name'][:], voltage_run['data'][:], voltage_run['pressure'][:])):
    mr, s = d
    b0 = np.quantile(s, 0.1)
    s -= b0
    color = plt.cm.plasma(i / len(voltage_run['name'][:])) # all colormaps are: plasma, viridis, inferno, magma, cividis
    sU.append(DatasetSpec(x=mr, y=s, label=f'$U_B = {U_B[i]:.2f} V$', line='-', marker='None', color=color, linewidth=0.5))
    s_smooth = savgol_filter(s, 15, 3)
    
    peaks, props = find_peaks(s_smooth, prominence=2, distance=30)
    
    mask2 = (mr[peaks] >= 25) & (mr[peaks] <= 35)
    peaks = peaks[mask2]
    props = {key: val[mask2] for key, val in props.items()}
    
    if peaks.size > 2:
        top2_idx = np.argsort(s_smooth[peaks])[-2:]
        peaks = peaks[top2_idx]
        props = {key: val[top2_idx] for key, val in props.items()}
    
    # sort peaks by m/z
    sort_idx = np.argsort(mr[peaks])
    peaks = peaks[sort_idx]
    props = {key: val[sort_idx] for key, val in props.items()}

    m_right = []
    m_left = []
    dm = []
    half_levels = []
    
    
    widths, height_eval, left_ips, right_ips = peak_widths(s_smooth, peaks, rel_height=1)
    
    for p in peaks:
        mR, mL, dm_val, half = fwhm(mr, s_smooth, p)
        m_right.append(mR)
        m_left.append(mL)
        dm.append(dm_val)
        half_levels.append(half)
    
    if props['prominences'][1] < 0.5*s[peaks[1]]:
        mR = np.interp(right_ips[1], np.arange(s_smooth.size), mr)
        mL = np.interp(left_ips[1], np.arange(s_smooth.size), mr)
        dm_val = mR - mL
        m_right[1]= mR
        m_left[1] = mL
        dm[1] = dm_val
        half_levels[1] = height_eval[1]


    mr_peaks = mr[peaks]
    signal_peaks = s[peaks]
    delta = np.array(dm)
    res = mr_peaks / delta

    s_peaks.append(DatasetSpec(x=mr_peaks, y=signal_peaks, line='None', marker='x', color='red', markersize=12, label=None if i else 'gefundene Peaks', 
                               axlines=[(half_levels[0], "h"), (half_levels[1], "h") ], axlines_intervals=[(m_left[0], m_right[0]), (m_left[1], m_right[1])], 
                               axlines_line='-.', axlines_color=color, axlines_label=None if i else [f'Δ(m/z) von jedem Peak'] ))
    
    e = 1.6e-19
    u = 1.66e-27

    N_N2 = nu * l *(mr_peaks[0] * u /( 2* U_B[i] * e))**0.5
    N_O2 = nu * l *(mr_peaks[1] * u /( 2* U_B[i] * e))**0.5
    
    print(f'U_B={U_B[i]:.2f}V: N_N2={N_N2:.2f}, N_O2={N_O2:.2f}, Δm={delta}, m={mr_peaks}')
    
    cycles.append((N_N2, N_O2))
    deltas.append(delta)
    masses.append(mr_peaks)


plot_data(
    sU + s_peaks,
    filename='Plots/voltage_run.pdf',
    title='Signalstärke bei verschiedenen Spannungen',
    xlabel='Massenladungszahl ($m/z$)',
    ylabel='Signalstärke (arb. Einheiten)',
    xlim=(10, 50),
    width=20,
    height=10,
    color_seed=42,
    plot=False,
)

s_cycles_N2 = DatasetSpec(x=[c[0] for c in cycles],y=[delta[0] for delta in deltas], label='N$_2^+$', marker='o')
s_cycles_O2 = DatasetSpec(x=[c[1] for c in cycles],y=[delta[1] for delta in deltas], label='O$_2^+$', marker='o')

plot_data(
    [s_cycles_N2, s_cycles_O2],
    filename='Plots/cycles.pdf',
    title='Oszillationen in Abhängigkeit von Δ(m/z)',
    ylabel='$\\Delta(m/z)$',
    xlabel='Anzahl der Oszillationen N',
    width=10,
    height=10,
    color_seed=42,
    plot=False,
)





data_volt = {
    'acceleration': U_B.tolist(),
    'mass_N2': [m[0] for m in masses],
    'mass_width_N2': [d[0] for d in deltas],
    'oscillations_N2': [c[0] for c in cycles],
    'mass_O2': [m[1] for m in masses],
    'mass_width_O2': [d[1] for d in deltas],
    'oscillations_O2': [c[1] for c in cycles],
}

headers_volt = {
    'acceleration': {'label': '{$U_B$ (V)}', 'repeat': False},
    'mass_N2': {'label': '{$m (u)$}', 'intermed': False},
    'mass_width_N2': {'label': '{$\\Delta m (u)$}', 'intermed': False},
    'oscillations_N2': {'label': '{$N$}', 'intermed': False},
    'mass_O2': {'label': '{$m (u)$}', 'intermed': False},
    'mass_width_O2': {'label': '{$\\Delta m (u)$}', 'intermed': False},
    'oscillations_O2': {'label': '{$N$}', 'intermed': False},
}

header_groups = [('', 1), ('N$_2^+$', 3), ('O$_2^+$', 3)]

print_standard_table(
    data_volt,
    headers=headers_volt,
    header_groups=header_groups,
    caption="Messungen der peaks von $N_2^+$ und $O_2^+$ von Luft bei verschiedenen Beschleunigungsspannungen.",
    label="tab:voltage",
    show=True
)







windows = [
    ("CH4/fragment", 14.5, 16.5),
    ("H2O",          17, 18.5),
    ("N2",           27, 28.5),
    ("O2",           29.5, 31.5),
    ("CO2",          43.5, 46),
]




sorting_res = np.argsort([int(name[5]) for name in resolution_run['name']])
resolution_run['name'] = [resolution_run['name'][i] for i in sorting_res]
resolution_run['data'] = [resolution_run['data'][i] for i in sorting_res]
resolution_run['pressure'] = [resolution_run['pressure'][i] for i in sorting_res]


# Store data per resolution: each entry is (resolution_value, mass_array, delta_m_array)
resolution_data = []
s_res, s_peaks = [], []

for i, (name, d, pressure) in enumerate(zip(resolution_run['name'], resolution_run['data'], resolution_run['pressure'])):
    mr, s = d
    b0 = np.quantile(s, 0.1)
    s -= b0
    
    s_smooth = savgol_filter(s, 31, 3)
    
    resid = s - s_smooth
    sigma = np.median(np.abs(resid - np.median(resid)))  # robust MAD sigma
    prom = 4 * sigma  # tune 4..10 depending on how aggressive you want it

    peaks_all, props = find_peaks(s_smooth, prominence=prom, distance=30)

    peaks_sel = []
    labels_sel = []

    for lab, lo, hi in windows:
        mask = (mr[peaks_all] >= lo) & (mr[peaks_all] <= hi)
        cand = peaks_all[mask]
        if cand.size == 0:
            continue

        # choose strongest candidate within this window
        p = cand[np.argmax(s_smooth[cand])]
        peaks_sel.append(p)
        labels_sel.append(lab)

    peaks = np.array(peaks_sel, dtype=int)
    
    # Filter props to match selected peaks
    selected_indices = [np.where(peaks_all == p)[0][0] for p in peaks]
    props = {key: val[selected_indices] for key, val in props.items()}

    # sort by m/z
    sort_idx = np.argsort(mr[peaks])
    peaks = peaks[sort_idx]
    props = {key: val[sort_idx] for key, val in props.items()}
    
    res_value = int(name[5])

    # Calculate widths for all peaks in this resolution
    m_right = []
    m_left = []
    dm = []
    half_levels = []
    
    widths, height_eval, left_ips, right_ips = peak_widths(s_smooth, peaks, rel_height=0.4 if res_value == 3 else 0.6)
    
    # Use FWHM for all peaks by default
    for p in peaks:
        mR, mL, dm_val, half = fwhm(mr, s_smooth, p)
        m_right.append(mR)
        m_left.append(mL)
        dm.append(dm_val)
        half_levels.append(half)
    
    # For res=2 and res=3, use prominence width for the first peak (index 0) only
    if res_value in [2, 3] and len(peaks) > 0:
        mR = np.interp(right_ips[0], np.arange(s_smooth.size), mr)
        mL = np.interp(left_ips[0], np.arange(s_smooth.size), mr)
        dm_val = mR - mL
        m_right[0] = mR
        m_left[0] = mL
        dm[0] = dm_val
        half_levels[0] = height_eval[0]

    
    mr_peaks = mr[peaks]
    dm_array = np.array(dm)
    signal_peaks = s[peaks]
    
    # Store all peaks for this resolution together
    resolution_data.append({
        'resolution': res_value,
        'masses': mr_peaks,
        'delta_m': dm_array,
        'name': name
    })
    
    print(f'Res={res_value}: m/z = {mr_peaks}, Δm = {dm_array}, Auflösung = {mr_peaks/dm_array}')
    
    color = plt.cm.viridis(res_value*10 / 100)
    s_res.append(DatasetSpec(x=mr, y=s, label=f'$Res = {res_value}$', line='-', marker='None', color=color, linewidth=0.5))
    s_peaks.append(DatasetSpec(x=mr_peaks, y=signal_peaks, line='None', marker='x', color='red', markersize=12, label=None if i else 'gefundene Peaks', 
                               axlines=[(half, 'h') for half in half_levels], axlines_intervals=[(m_left[j], m_right[j]) for j in range(len(peaks))],
                               axlines_line='--', axlines_color=color, axlines_label=None if i else [f'Δ(m/z) von jedem Peak'] ))

plot_data(
    s_res + s_peaks,
    filename='Plots/resolution_run.pdf',
    title='Signalstärke bei verschiedenen Genauigkeiten',
    xlabel='Massenladungszahl ($m/z$)',
    ylabel='Signalstärke (arb. Einheiten)',
    xlim=(10, 50),
    width=20,
    height=10,
    color_seed=42,
    plot=False,
)


s_mass = []
fit_results_by_res = {}  # Store fit results for table

for data in resolution_data:
    res_value = data['resolution']
    masses_arr = data['masses']
    delta_m_arr = data['delta_m']
    
    color = plt.cm.viridis(res_value * 10 / 100)
    
    # Perform linear fit for this resolution's data
    if len(masses_arr) > 1:  # Need at least 2 points for a fit
        results = lmfit(model='linear', xdata=masses_arr, ydata=delta_m_arr)
        fit_mr = np.linspace(masses_arr.min(), masses_arr.max(), 100)
        fit_y = results.eval(x=fit_mr)
        
        # Store fit parameters for table
        residuals = results.residual
        sigma = np.std(residuals)
        
        fit_results_by_res[res_value] = {
            'masses': masses_arr,
            'delta_m': delta_m_arr,
            'intercept': results.params['a'].value,
            'intercept_err': results.params['a'].stderr if results.params['a'].stderr else 0,
            'slope': results.params['b'].value,
            'slope_err': results.params['b'].stderr if results.params['b'].stderr else 0,
            'sigma': sigma,
            'r_squared': 1 - results.residual.var() / np.var(delta_m_arr) if np.var(delta_m_arr) > 0 else 0
        }
        
        s_mass.append(DatasetSpec(
            x=masses_arr, 
            y=delta_m_arr, 
            marker='o', 
            color=color, 
            fit_x=fit_mr, 
            fit_y=fit_y, 
            fit_label=f'Res={res_value} Fit', 
            label=f'Res={res_value}'
        ))
    else:
        # Just plot the single point without fit
        fit_results_by_res[res_value] = {
            'masses': masses_arr,
            'delta_m': delta_m_arr,
            'intercept': None,
            'intercept_err': None,
            'slope': None,
            'slope_err': None,
            'sigma': None,
            'r_squared': None
        }
        s_mass.append(DatasetSpec(
            x=masses_arr, 
            y=delta_m_arr, 
            marker='o', 
            color=color, 
            label=f'Res={res_value}'
        ))

fig, ax = plot_data(
    s_mass,
    title='Massenauflösung bei verschiedenen Genauigkeiten',
    xlabel='Massen $m/z$',
    ylabel='$\\Delta m/z$',
    width=20,
    height=20,
    color_seed=42,
    plot='figure',
)

ax.xaxis.set_major_locator(MultipleLocator(2))
plt.savefig('Plots/mass_resolution.pdf')


# Prepare data for resolution table (peaks only)
# Find max number of peaks across all resolutions
max_peaks = max(len(fit_results_by_res[res]['masses']) for res in fit_results_by_res)

# Build data dictionary for the peaks table
data_res = {}
for res in sorted(fit_results_by_res.keys()):
    fit_data = fit_results_by_res[res]
    masses = fit_data['masses'].tolist()
    delta_m = fit_data['delta_m'].tolist()
    
    # Pad with empty strings to match max_peaks length
    masses_padded = masses + [''] * (max_peaks - len(masses))
    delta_m_padded = delta_m + [''] * (max_peaks - len(delta_m))
    
    data_res[f'm_{res}'] = masses_padded
    data_res[f'dm_{res}'] = delta_m_padded

# Build headers dictionary for peaks table
headers_res = {}
for res in sorted(fit_results_by_res.keys()):
    headers_res[f'm_{res}'] = {'label': '{$m/z$}', 'intermed': False}
    headers_res[f'dm_{res}'] = {'label': '{$\\Delta m/z$}', 'intermed': False}

# Header groups: (name, num_columns)
header_groups = [(f'{res}', 2) for res in sorted(fit_results_by_res.keys())]

print_standard_table(
    data_res,
    headers=headers_res,
    header_groups=header_groups,
    caption="Gemessene Peaks bei verschiedenen Auflösungen.",
    label="tab:resolution_peaks",
    show=True
)

# Second table: Fit parameters (intercept, slope, errors, sigma, and R²)
data_fit = {
    'resolution': sorted(fit_results_by_res.keys()),
    'intercept': [fit_results_by_res[res]['intercept'] for res in sorted(fit_results_by_res.keys())],
    'intercept_err': [fit_results_by_res[res]['intercept_err'] for res in sorted(fit_results_by_res.keys())],
    'slope': [fit_results_by_res[res]['slope'] for res in sorted(fit_results_by_res.keys())],
    'slope_err': [fit_results_by_res[res]['slope_err'] for res in sorted(fit_results_by_res.keys())],
    'sigma': [fit_results_by_res[res]['sigma'] for res in sorted(fit_results_by_res.keys())],
    'r_squared': [fit_results_by_res[res]['r_squared'] for res in sorted(fit_results_by_res.keys())],
}

headers_fit = {
    'resolution': {'label': '{Auflösung}', 'intermed': True, 'round': False},
    'intercept': {'label': '{$a$}', 'intermed': True, 'err': data_fit['intercept_err']},
    'slope': {'label': '{$b$}', 'intermed': True, 'err': data_fit['slope_err']},
    'sigma': {'label': '{$\\sigma_{res}$}', 'intermed': True},
    'r_squared': {'label': '{$R^2$}', 'intermed': True},
}

print_standard_table(
    data_fit,
    headers=headers_fit,
    caption="Lineare Fit-Parameter der Beziehung $\Delta m = a + b\,m$ für verschiedene Auflösungen. Hier entspricht $b$ der Steigung (in erster Näherung $\Delta m/m$), $a$ ist ein massenunabhängiger Offset.",
    label="tab:resolution_fit",
    show=True
)





def shift_to_28(mr, y_smooth, lo=27.4, hi=28.6, target=28.0):
    """Shift mr so that the local maximum in [lo, hi] lands at target."""
    w = (mr >= lo) & (mr <= hi)
    if not np.any(w):
        return mr, 0.0

    idx = np.argmax(y_smooth[w])
    m_peak = mr[w][idx]
    offset = m_peak - target
    return mr - offset, offset





data={
    'Aceton': {
        "mz":  np.array([14, 15, 26, 27, 28, 29, 37, 38, 39, 40, 41, 42, 43, 44, 57, 58, 59]),
        "s": np.array([2.9, 23.1, 3.5, 5.7, 1.2, 3.1, 1.8, 2.2, 4.2, 1.0, 2.0, 9.1, 100.0, 3.4, 1.7, 63.8, 3.1]),
        "mask": (10, 60)
    },
    'Ethanol': {
        "mz": np.array([14, 15, 19, 26, 27, 28, 29, 30, 31, 32, 41, 42, 43, 44, 45, 46]),
        "s": np.array([1.4, 3.4, 2.3, 4.9, 17.7, 4.2, 12.0, 5.0, 100.0, 1.4, 1.0, 3.4, 9.9, 1.0, 57.3, 24.6]),
        "mask": (10, 50)
    },
    'Air': {
        "mz": np.array([14, 16, 20, 28, 29, 32, 34, 36, 40, 44]),
        "s": np.array([7.200, 2.635, 0.154, 100.0, 0.800, 23.068, 0.0923, 0.00431, 1.435, 0.0592]),
        "mask": (10, 50)
    },
    'Ar': {
        "mz": np.array([40, 20]),
        "s": np.array([100.0, 10.7]),
        "mask": (10, 50)
    },
}


s_mol = []
for name, d, pressure in zip(molecules_run['name'], molecules_run['data'], molecules_run['pressure']):
    mr, s = d
    name_mol = name[:-2]
    mask = (mr >= data[name_mol]['mask'][0]) & (mr <= data[name_mol]['mask'][1])
    mr = mr[mask]
    s = s[mask]
    
    b0 = np.quantile(s, 0.3)
    s -= b0
    mask2 = s > 0
    s = s[mask2]
    mr = mr[mask2]
    s_filtered = savgol_filter(s, 31, 3)
    
    resid = s - s_filtered
    sigma = np.median(np.abs(resid - np.median(resid))) 
    prom = 4 * sigma  

    peaks, props = find_peaks(s_filtered, prominence=prom, distance=30)

    mr_peaks = mr[peaks]
    s_filtered_peaks = s_filtered[peaks]
    
    offset = mr_peaks[np.argmax(s_filtered_peaks)] - 28
    
    mr, offset = shift_to_28(mr, s_filtered)
    mr_peaks -= offset
    
    s_mol= DatasetSpec(x=mr, y=s, label=f"${name_mol}$", line='-', marker='None',color='tab:green', fit_y=s_filtered, fit_x=mr, fit_label=f'${name_mol}$ Smooth', fit_color='tab:blue', linewidth=0.5)
    s_peaks = DatasetSpec(x=mr_peaks, y=s_filtered_peaks, label=f'${name_mol}$ Peaks', line='None', marker='x', color='tab:red', markersize=24)
    s_peaks_bar = DatasetSpec(x=np.rint(mr_peaks).astype(int)+0.1, y=s_filtered_peaks, label=f'${name_mol}$ Peaks', plot_style='bar', barwidth=0.2, color='tab:red')

    fig, ax = plot_data(
        [s_peaks_bar, s_mol, s_peaks],
        #filename=f'Plots/{name}.pdf',
        title=f'Signalstärke von {name.replace("_", " ")} und gefundene Peaks',
        xlabel='Massenladungszahl ($m/z$)',
        ylabel='Signalstärke (arb. Einheiten)',
        width=20,
        xlim=data[name_mol]['mask'],
        height=10,
        color_seed=33,
        plot='figure',
    )
    ax.xaxis.set_major_locator(MultipleLocator(2))
    plt.savefig(f'Plots/{name}.pdf')
    
    s_filtered_peaks /= max(s_filtered_peaks)/100
    
    s_peaks_bar = DatasetSpec(x=np.rint(mr_peaks).astype(int)+0.1, y=s_filtered_peaks, label=f'${name_mol}$ Peaks', plot_style='bar', barwidth=0.2, color='tab:red')
    s_theory = DatasetSpec(x=data[name_mol]['mz']-0.1, y=data[name_mol]['s'], label=f'${name_mol}$ Theorie', color='tab:purple', plot_style='bar', barwidth=0.2)

    
    
    fig, ax = plot_data(
        [s_theory, s_peaks_bar],
        #filename=f'Plots/{name}_spectrum.pdf',
        title=f'Massenspektrum von {name.replace("_", " ")} im Vergleich zur Theorie',
        xlabel='Massenladungszahl ($m/z$)',
        ylabel='Signalstärke (arb. Einheiten)',
        xlim=data[name_mol]['mask'],
        width=20,
        height=10,
        color_seed=33,
        plot='figure',
    )
    
    ax.xaxis.set_major_locator(MultipleLocator(2))
    plt.savefig(f'Plots/{name}_spectrum.pdf')

