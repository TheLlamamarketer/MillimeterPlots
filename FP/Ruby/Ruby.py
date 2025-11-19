import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.signal import find_peaks, peak_widths
import scipy.constants as consts

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Functions.plotting import DatasetSpec, plot_data
from Functions.tables import *
from Functions.help import *

source_dir = Path('FP/Ruby/Data')

# Calibration parameters determined from Hg lines
Step_corrected = 0.59796398
Start_offset = 350 - 346.666132
B_val = 765.0 # cm^-1

Hg_lines = np.array([365.0153, 404.6563, 435.8328, 253.6517*2, 546.0735, 576.9598, 579.0663, 296.7280*2, 312.5668*2, 313.16935000*2])


data: dict[str, np.ndarray] = {}


for file in sorted(source_dir.iterdir()):
    if not file.is_file() or file.stem.lower() == '9_diff_b2':
        continue
    
    # Read first 3 lines separately
    with open(file, 'r') as f:
        start, step, direction = [float(f.readline().strip()) for _ in range(3)]

    if step == 0.6:
        step = Step_corrected
    elif step == 1.0:
        step = Step_corrected * 10/6
    start -= Start_offset
    
    raw = pd.read_csv(
        file, header=None, decimal=".", sep=r"\s+", engine="python", skiprows=3).to_numpy(dtype=float)

    t = raw[:, 0]
    A = raw[:, 1]
    
    key = file.stem[2:] if not file.stem.startswith('Hg') else file.stem
    
    data[key] = {
        't': t, 
        'A': A, 
        'header': [start, step, direction],
        'lambda': start + direction * step * t
    }

# --------------------------------------------------------------------------------
# --- Analyze Hg lines for calibration -------------------------------------------
# --------------------------------------------------------------------------------

data['Hg_lines']['A'] = -(data['Hg_lines']['A'] + np.min(-data['Hg_lines']['A']))          # invert and shift to zero baseline as we have emission lines

A_Hg = data['Hg_lines']['A']
A_Hg = A_Hg / np.max(A_Hg)  # normalize to max
data['Hg_lines']['A'] = A_Hg

lam_Hg = data['Hg_lines']['lambda']


# find peaks
peaks_hg, props_hg = find_peaks(A_Hg,  height=0.012)
widths_hg = peak_widths(A_Hg, peaks_hg, rel_height=0.21)


results_Hg_y = np.array([])
t_gauss = np.array([])

line_peaks: list[tuple[float, float]] = [] 

for i, peak_i in enumerate(peaks_hg):
    widths = widths_hg[0][i] * 0.1
    width_heights = widths_hg[1][i]
    left_ips = int(widths_hg[2][i])
    right_ips = int(widths_hg[3][i])
    
    #expand = int(widths / 2)
    expand = 0
    fit_left = max(0, left_ips - expand)
    fit_right = min(len(lam_Hg) - 1, right_ips + expand)

    mask = (lam_Hg >= lam_Hg[fit_left]) & (lam_Hg <= lam_Hg[fit_right])
    
    lam_per_peak = lam_Hg[mask]
    A_per_peak = A_Hg[mask]
    
    result_Hg = lmfit(
        xdata= lam_per_peak,
        ydata= A_per_peak,
        model="gaussian",
        initial_params={'c': widths/2.355, 'b': lam_Hg[peak_i]} # a * np.exp(-((x - b) / c)**2 / 2) + d
    )
    
    #lam_gauss = np.linspace(lam_per_peak[0], lam_per_peak[-1], 100)
    #results_Hg_y = np.append(results_Hg_y, result_Hg.eval(x=lam_gauss), axis=0)   
    
    results_Hg_y = np.append(results_Hg_y, [result_Hg], axis=0)
    b_val = result_Hg.params['b'].value
    
    line_peaks.append((b_val, props_hg['peak_heights'][i])) # or result_Hg.eval(x=b_val) for amplitude


hg_start, hg_step, hg_direction = data["Hg_lines"]["header"]
lambda_peaks = np.array([peak[0] for peak in line_peaks])
t_hg_peaks = (lambda_peaks - hg_start) / (hg_direction * hg_step)
t_hg_peaks = data["Hg_lines"]["t"][peaks_hg]


data_hg = {
    'linelist': Hg_lines,
    'fitted': lambda_peaks,
    'dfitted': [result_Hg.params['b'].stderr for result_Hg in results_Hg_y],
    'diff': lambda_peaks - Hg_lines,
}


headers_hg = {
    'linelist':     {'label': '{$Hg_{lines}/nm$}', 'intermed': True },
    'fitted':       {'label': '{$Hg_{fitted}/nm$}', 'intermed': True, 'err': data_hg['dfitted']},
    'diff':         {'label': '{$\\Delta Hg/nm$}', 'intermed': True},
}

print_standard_table(
    data_hg,
    headers_hg,
    caption="Fitted Hg line positions and their differences to the known literature values.",
    label="tab:ruby_hg_lines",
    show=True
)

results_line_hg = lmfit(xdata=t_hg_peaks, ydata=Hg_lines, model="linear")

print_line = True


if print_line:
    print(results_line_hg.fit_report())
    print(f"a = {results_line_hg.params['a'].value:.3f} ± {results_line_hg.params['a'].stderr:.3f}, b = {results_line_hg.params['b'].value:.3f} ± {results_line_hg.params['b'].stderr:.3f}")


# --------------------------------------------------------------------------------
# --- Plotting the Hg Lines ------------------------------------------------------
# --------------------------------------------------------------------------------

s_hg = DatasetSpec(
    x=lam_Hg,
    y=A_Hg,
    label="Hg_lines",
    marker="None",
    line="-",
    axlines=[(line, "|") for line in Hg_lines],
    axlines_color="green",
    axlines_label="Known Hg lines",
    axlines_intervals=[(-0.05, peak) for peak in A_Hg[peaks_hg]],
)

s_peaks = DatasetSpec(
    x=lam_Hg[peaks_hg],
    y=A_Hg[peaks_hg],
    label="Peaks from find_peaks",
    marker="x",
    line="None",
)
s_gauss = DatasetSpec(
    x=lambda_peaks,
    y=np.array([peak[1] for peak in line_peaks]),
    label=f'Peaks from Gaussian Fit',
    marker='.',
    line='None',
)

plot_data(
    [s_hg, s_peaks, s_gauss],
    title=f'Ruby Data: Hg',
    xlabel='Wavelength (nm)',
    ylabel='Amplitude (normalized)',
    filename=f'Plots/Hg_lines.pdf',
    height=15,
    color_seed=89,
    plot=False
)


# --------------------------------------------------------------------------------
# --- Plotting all other data ----------------------------------------------------
# --------------------------------------------------------------------------------
# Build specs for all non-Hg datasets
names = ['diff', 'Xe', 'Ruby']

data['Ruby_N']['A'] *= 1.02669405

for name in names:

    lamB, yB = np.asarray(data[f'{name}_B']['lambda'],float), np.asarray(data[f'{name}_B']['A'],float)
    lamN, yN = np.asarray(data[f'{name}_N']['lambda'],float), np.asarray(data[f'{name}_N']['A'],float)
    # sort
    si = np.argsort(lamB); lamB, yB = lamB[si], yB[si]
    si = np.argsort(lamN); lamN, yN = lamN[si], yN[si]

    # objective: shift B by off and compute MSE on overlap (interpolating B onto N grid)
    def mse_off(off):
        lb = lamB + off
        lo = max(lb.min(), lamN.min()); hi = min(lb.max(), lamN.max())
        if hi <= lo: return 1e12
        mask = (lamN >= lo) & (lamN <= hi)
        if mask.sum() < 4: return 1e12
        yBinterp = np.interp(lamN[mask], lb, yB, left=np.nan, right=np.nan)
        valid = ~np.isnan(yBinterp)
        if valid.sum() < 4: return 1e12
        r = yN[mask][valid] - yBinterp[valid]
        return float((r*r).mean())

    span = min(lamN.max()-lamN.min(), lamB.max()-lamB.min())
    b = min(span/2.0, 50.0)

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(mse_off, bounds=(-b,b), method='bounded')
    best = float(res.x)

    data[f'{name}_B']['lambda'] = data[f'{name}_B']['lambda'] + best
    print(f"Applied lambda offset to {name}_B: {best:.6g}")

#data['Ruby_B']['lambda'] += 0.6

specs_by_key: dict[str, DatasetSpec] = {}
for key in sorted(data.keys()):
    if key == 'Hg_lines' or key == 'Xe_Basis':
        continue
    prefix = key.split('_')[0]
    bkey = f"{prefix}_B"
    is_ruby = prefix == 'Ruby'

    if bkey in data:
        # Use baseline (minimum) from the corresponding _B dataset
        base_min = np.min(data[bkey]['A']) if is_ruby else np.min(-data[bkey]['A'])
        y = (data[key]['A'] - base_min) if is_ruby else (-data[key]['A'] - base_min)
    else:
        # Fallback to per-dataset minimum if no _B reference exists
        y = (data[key]['A'] - np.min(data[key]['A'])) if is_ruby else (-data[key]['A'] - np.min(-data[key]['A']))
    

    specs_by_key[key] = DatasetSpec(
        x=data[key]['lambda'],
        y=y,
        label=key,
        marker='None',
        line='-',
    )

# Group datasets by prefix
grouped: dict[str, list[DatasetSpec]] = {name: [] for name in names}
for key, spec in specs_by_key.items():
    for name in names:
        if key.startswith(name):
            grouped[name].append(spec)
            break

# Plot each group
for k, specs in grouped.items():
    if not specs:
        continue
    plot_data(
        specs,
        title=f'Ruby Data: N and B lines with {k} Spectrum',
        xlabel='Wavelength (nm)',
        ylabel='Amplitude',
        filename=f'Plots/N_B_lines_{k}.pdf',
        log_scale= (None, 10),
        height=15,
        plot=False
    )


s_diff = DatasetSpec(x=data['diff_B']['lambda'], y=np.log10(1/np.clip(-data['diff_B']['A'], 1e-12, None)), label='Ruby_B / Xe_B (transmittance)', marker='None', line='-')

#results_diff = lmfit(xdata=lam_B, ydata=np.log10(1/T_B), model="gaussian", initial_params={'b': 408, 'c': 1.0})



plot_data(
    [s_diff],
    title=f'Ruby Data: Difference',
    xlabel='Wavelength (nm)',
    ylabel='Absorbance (A)',
    filename=f'Plots/Difference.pdf',
    log_scale=(None, None),
    height=15,
    plot=False
)



# Compute transmittance Ruby_B / Xe_B using interpolation onto Ruby_B wavelength grid
def compute_ratio(lam_num, I, lam_den, I_0, eps=1e-12, ngrid=None):
    lam_num = np.asarray(lam_num, dtype=float)
    I = np.asarray(I, dtype=float)
    lam_den = np.asarray(lam_den, dtype=float)
    I_0 = np.asarray(I_0, dtype=float)

    lo = max(lam_num.min(), lam_den.min())
    hi = min(lam_num.max(), lam_den.max())
    if hi <= lo:
        return np.array([], dtype=float), np.array([], dtype=float)

    n = int(ngrid or max(len(lam_num), len(lam_den)))
    grid = np.linspace(lo, hi, n)
    num_i = np.interp(grid, lam_num, I)
    den_i = np.interp(grid, lam_den, I_0)
    mask = np.abs(den_i) > eps
    if mask.sum() == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    T = num_i[mask] / den_i[mask]
    T = np.clip(T, eps, None)
    # debug info (small, helpful when running locally)
    print(f"compute_ratio: grid_points={n}, valid_points={mask.sum()}, overlap=({lo:.3f},{hi:.3f})")
    return grid[mask], T

lam_B, T_B = compute_ratio(data['Ruby_B']['lambda'], -data['Ruby_B']['A'], data['Xe_B']['lambda'], data['Xe_B']['A'], eps=1e-6)

initial_T_1 = 406.60
initial_T_2 = 561.46
mask_408 = (lam_B >= initial_T_1 - 15) & (lam_B <= initial_T_1 + 15)
mask_560 = (lam_B >= initial_T_2 - 25) & (lam_B <= initial_T_2 + 25)

results_diff_T_1 = lmfit(xdata=lam_B[mask_408], ydata=np.log10(1/T_B)[mask_408], model="gaussian", initial_params={'b': initial_T_1, 'd': np.min(np.log10(1/T_B)) })
results_diff_T_2 = lmfit(xdata=lam_B[mask_560], ydata=np.log10(1/T_B)[mask_560], model="gaussian", initial_params={'b': initial_T_2, 'd': np.min(np.log10(1/T_B)) })

fit_x_T_1 = lam_B[mask_408]
fit_y_T_1 = results_diff_T_1.eval(x=fit_x_T_1)
fit_x_T_2 = lam_B[mask_560]
fit_y_T_2 = results_diff_T_2.eval(x=fit_x_T_2)

fit_x = np.concatenate([fit_x_T_1, [(fit_x_T_1[-1] + fit_x_T_2[0])/2], fit_x_T_2])
fit_y = np.concatenate([fit_y_T_1, [np.nan], fit_y_T_2])

s_B = DatasetSpec(x=lam_B, y=np.log10(1/T_B), label='Ruby_B / Xe_B', marker='None', line='-', fit_y=fit_y, fit_x=fit_x, fit_line='-', fit_label='Gaussian Fit 408nm', fit_color='cyan')

print(results_diff_T_1.fit_report())
print(results_diff_T_2.fit_report())

const = 10**7 / B_val

print(f"Resulting centers: ^4 T_1 fit = {results_diff_T_1.params['b'].value:.2f} ± {results_diff_T_1.params['b'].stderr:.2f}, ^4 T_2 fit = {results_diff_T_2.params['b'].value:.2f} ± {results_diff_T_2.params['b'].stderr:.2f}")
print(f"Resulting normed Emergies by B: ^4 T_1 = {const/ results_diff_T_1.params['b'].value:.2f}, ^4 T_2 = {const/ results_diff_T_2.params['b'].value:.2f} with ratio {results_diff_T_2.params['b'].value / results_diff_T_1.params['b'].value:.4f}")


plot_data(
    [s_B],
    title=f'Ruby Data: Absorbance Ruby / Xe Spectra',
    xlabel='Wavelength (nm)',
    ylabel='Absorbance (A)',
    filename=f'Plots/Abs_Ruby_Xe.pdf',
    log_scale=(None, None),
    height=15,
    plot=True
)
