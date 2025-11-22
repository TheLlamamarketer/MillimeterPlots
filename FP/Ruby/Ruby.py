import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import make_splrep, interp1d
from scipy.optimize import fsolve, minimize_scalar
from SVG_ST import spline_A2, spline_T2, spline_T1



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
B_val = 765.0 # cm^-1  or 638.0

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
    'original': t_hg_peaks*0.6 + 350,
}


headers_hg = {
    'linelist':     {'label': '{${Hg}_{lines}/nm$}', 'intermed': True },
    'fitted':       {'label': '{${Hg}_{fit}/nm$}', 'intermed': True, 'err': data_hg['dfitted']},
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




names = [ 'Xe', 'Ruby'] #'diff',

data['Ruby_N']['A'] *= 1.02669405










data['diff_B']['lambda'] -= 0.185
data['diff_N']['lambda'] -= 0.185




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

    
    data[key]['A'] = data[key]['A'] if is_ruby else -data[key]['A']
    specs_by_key[key] = DatasetSpec(
        x=data[key]['lambda'],
        y=data[key]['A'],
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
        height=15,
        plot=False
    )













# --------------------------------------------------------------------------------
# --- Compute transmittance Ruby_B / Xe_B and fit peaks --------------------------
# --------------------------------------------------------------------------------



def compute_ratio(lam_num, I, lam_den, I_0, eps=1e-12, ngrid=None):
    lam_num = np.asarray(lam_num, dtype=float)
    I = np.asarray(I, dtype=float)
    lam_den = np.asarray(lam_den, dtype=float)
    I_0 = np.asarray(I_0, dtype=float)
    
    lam_num_sort = np.argsort(lam_num)
    lam_num = lam_num[lam_num_sort]
    I = I[lam_num_sort]
    
    lam_den_sort = np.argsort(lam_den)
    lam_den = lam_den[lam_den_sort]
    I_0 = I_0[lam_den_sort]


    lo = max(lam_num.min(), lam_den.min())
    hi = min(lam_num.max(), lam_den.max())
    if hi <= lo:
        return np.array([], dtype=float), np.array([], dtype=float)

    n = int(ngrid or max(len(lam_num), len(lam_den)))
    grid = np.linspace(lo, hi, n)
    num_i = make_splrep(lam_num, I, s=0.0)(grid)
    den_i = make_splrep(lam_den, I_0, s=0.0)(grid)
    mask = np.abs(den_i) > eps
    if mask.sum() == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    T = num_i[mask] / den_i[mask]
    T = np.clip(T, eps, None)
    print(f"compute_ratio: grid_points={n}, valid_points={mask.sum()}, overlap=({lo:.3f},{hi:.3f})")
    return grid[mask], T







plot_data(
    [DatasetSpec(x=data['diff_B']['lambda'], y=data['diff_B']['A'], label='diff_B', marker='None', line='-'),
     DatasetSpec(x=data['Xe_B']['lambda'], y=data['Xe_B']['A'], label='Xe_B', marker='None', line='-')],
    title=f'Ruby Data: Difference',
    xlabel='Wavelength (nm)',
    ylabel='Intensity I',
    height=15,
    plot=False
)







lam_diff_B, T_diff_B = compute_ratio(data['diff_B']['lambda'], data['diff_B']['A'], data['Xe_B']['lambda'], data['Xe_B']['A'])
print(lam_diff_B, T_diff_B)
s_diff = DatasetSpec(x=lam_diff_B, y=T_diff_B, label='diff_B / Xe_B (Absorbance)', marker='None', line='-') #np.log(1/(1-T_diff_B))

#results_diff = lmfit(xdata=lam_diff_B, ydata=np.log10(1/T_diff_B), model="gaussian", initial_params={'b': 408, 'c': 1.0})


plot_data(
    [s_diff],
    title=f'Ruby Data: Difference',
    xlabel='Wavelength (nm)',
    ylabel='Absorbance (A)',
    filename=f'Plots/Difference.pdf',
    height=15,
    plot=False
)



# Compute transmittance Ruby_B / Xe_B using interpolation onto Ruby_B wavelength grid


lam_B, T_B = compute_ratio(data['Ruby_B']['lambda'], data['Ruby_B']['A'], data['Xe_B']['lambda'], data['Xe_B']['A'], eps=1e-6)

initial_T_1 = 408.0
initial_T_2 = 558.0
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

print(f"Resulting centers: ^4 T_1 fit = {results_diff_T_1.params['b'].value:.2f} ± {results_diff_T_1.params['b'].stderr:.2f}, ^4 T_2 fit = {results_diff_T_2.params['b'].value:.2f} ± {results_diff_T_2.params['b'].value:.2f}")
print(f"Resulting normed Emergies by B: ^4 T_1 = {const/ results_diff_T_1.params['b'].value:.3f} ± {const * results_diff_T_1.params['b'].stderr / (results_diff_T_1.params['b'].value**2):.3f}, ^4 T_2 = {const/ results_diff_T_2.params['b'].value:.3f}  ± {const * results_diff_T_2.params['b'].value / (results_diff_T_2.params['b'].value**2):.3f} with ratio {results_diff_T_2.params['b'].value / results_diff_T_1.params['b'].value:.4f}")


plot_data(
    [s_B],
    title=f'Ruby Data: Absorbance Ruby / Xe Spectra',
    xlabel='Wavelength (nm)',
    ylabel='Absorbance (A)',
    filename=f'Plots/Abs_Ruby_Xe.pdf',
    height=15,
    plot=False
)

def build_level(lbl, spline, fit_result, const, x, root_guess=2.0):
    # energy from fit
    b = fit_result.params['b'].value
    b_err = fit_result.params['b'].stderr or 0.0
    E_fit = const / b
    dE_fit = const * b_err / b**2  # uncertainty propagation

    # invert spline to get Dq at fitted energy
    Dq_fit = fsolve(lambda Dq: spline(Dq) - E_fit, x0=root_guess)[0]

    spec = DatasetSpec(
        x=x,
        y=spline(x),
        label=lbl,
        line='-',
        marker='None',
        axlines=[(E_fit, '-'), (E_fit + dE_fit, '-'), (E_fit - dE_fit, '-'), (Dq_fit, '|')],
        axlines_label=[f'$E_{{{lbl[-2]}}}$ Fit', '', '', '$\\approx 1 Dq$'],
        axlines_intervals=[(), (), (), (0, E_fit)],
        axlines_line=['--', '--', '--', '-']
    )
    return E_fit, dE_fit, Dq_fit, spline, spec

x = np.linspace(0, 4.154, 1000)
E_T1, dE_T1, Dq_T1, spline_T1, s_T1 = build_level('$^4 T_1$', spline_T1, results_diff_T_1, const, x)
E_T2, dE_T2, Dq_T2, spline_T2, s_T2 = build_level('$^4 T_2$', spline_T2, results_diff_T_2, const, x)

print(f"Fitted Energy Levels by B: E_T1 = {E_T1:.3f} ± {dE_T1:.3f} at Dq/B = {Dq_T1:.3f}, E_T2 = {E_T2:.3f} ± {dE_T2:.3f} at Dq/B = {Dq_T2:.3f}, with ratio E_T2 / E_T1 = {E_T2 / E_T1:.4f}")
print(f"Resulting 10 Dq from fits: 10 Dq = {10* Dq_T1 * B_val:.2f} cm^-1 or {10 * Dq_T1 * B_val * 1.23981e-4:.4f} eV from T1, 10 Dq = {10 * Dq_T2 * B_val:.2f} cm^-1 or {10 * Dq_T2 * B_val * 1.23981e-4:.4f} eV from T2")


plot_data(
    [s_T1, s_T2],
    title=f'Ruby Data: Energy Levels',
    xlabel='Dq / B',
    ylabel='Energy E / B',
    width=10,
    height=17,
    filename=f'Plots/Tanabe_Sugano_4T.pdf',
    plot=False
)





# --------------------------------------------------------------------------------
# --- Computing the narrow regions around 680nm and 720nm ----------------------
# --------------------------------------------------------------------------------
data['diff_N']['lambda'] -= 9.83 + 0.296
data['Xe_N']['lambda'] -= 0.31




    
    


lam_Xe_N = data['Xe_N']['lambda'][(670 <= data['Xe_N']['lambda']) & (data['Xe_N']['lambda'] <= 715)]
lam_Ruby_N = data['Ruby_N']['lambda'][(670 <= data['Ruby_N']['lambda']) & (data['Ruby_N']['lambda'] <= 715)]
lam_diff_N = data['diff_N']['lambda'][(670 <= data['diff_N']['lambda']) & (data['diff_N']['lambda'] <= 715)]

I_Xe_N = data['Xe_N']['A'][(670 <= data['Xe_N']['lambda']) & (data['Xe_N']['lambda'] <= 715)]
I_Ruby_N = data['Ruby_N']['A'][(670 <= data['Ruby_N']['lambda']) & (data['Ruby_N']['lambda'] <= 715)]
I_diff_N = data['diff_N']['A'][(670 <= data['diff_N']['lambda']) & (data['diff_N']['lambda'] <= 715)]

lam_Xe_N = np.asarray(lam_Xe_N, dtype=float)
lam_Ruby_N = np.asarray(lam_Ruby_N, dtype=float)
lam_diff_N = np.asarray(lam_diff_N, dtype=float)

I_Xe_N = np.asarray(I_Xe_N, dtype=float)
I_Ruby_N = np.asarray(I_Ruby_N, dtype=float)
I_diff_N = np.asarray(I_diff_N, dtype=float)


lam_Xe_sort = np.argsort(lam_Xe_N)
lam_Xe_N = lam_Xe_N[lam_Xe_sort]
I_Xe_N = I_Xe_N[lam_Xe_sort]

lam_Ruby_sort = np.argsort(lam_Ruby_N)
lam_Ruby_N = lam_Ruby_N[lam_Ruby_sort]
I_Ruby_N = I_Ruby_N[lam_Ruby_sort]

lam_diff_sort = np.argsort(lam_diff_N)
lam_diff_N = lam_diff_N[lam_diff_sort]
I_diff_N = I_diff_N[lam_diff_sort]

n = int(max(len(lam_Xe_N), len(lam_Ruby_N)))
grid = np.linspace(670, 715, n)
Xe_N = make_splrep(lam_Xe_N, I_Xe_N, s=0.0)(grid)
Ruby_N = make_splrep(lam_Ruby_N, I_Ruby_N, s=0.0)(grid)
Diff_N = make_splrep(lam_diff_N, I_diff_N, s=0.0)(grid)




plot_data(
    [DatasetSpec(x=grid, y=Diff_N, label='diff_B', marker='None', line='-'),
     DatasetSpec(x=grid,   y=Xe_N, label='Xe_B', marker='None', line='-'),
     DatasetSpec(x=grid, y=Ruby_N, label='Ruby_B', marker='None', line='-'),
     DatasetSpec(x=grid, y=-Ruby_N+Xe_N, label='Difference', marker='None', line='--')],
    title=f'Ruby Data: Difference',
    xlabel='Wavelength (nm)',
    ylabel='Intensity I',
    height=15,
    plot=True
)



