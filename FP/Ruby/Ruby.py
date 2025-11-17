import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

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

data['Hg_lines']['A'] = -1 * (data['Hg_lines']['A'] + np.min(-data['Hg_lines']['A'])) # invert and shift to zero baseline as we have emission lines

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
    'diff':         {'label': '{$\Delta Hg/nm$}', 'intermed': True},
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
    plot=True
)


# --------------------------------------------------------------------------------
# --- Plotting all other data ----------------------------------------------------
# --------------------------------------------------------------------------------
# Build specs for all non-Hg datasets
names = ['diff', 'Xe', 'Ruby']

specs_by_key: dict[str, DatasetSpec] = {}
for key in sorted(data.keys()):
    if key == 'Hg_lines':
        continue
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
        plot=True
    )


