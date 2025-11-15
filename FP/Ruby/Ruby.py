from re import A
import sys
from pathlib import Path
from turtle import width
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

# file has structure: (t, Amplitude)

data: dict[str, np.ndarray] = {}
for file in sorted(source_dir.iterdir()):
    if not file.is_file():
        continue
    
    # Read first 3 lines separately
    with open(file, 'r') as f:
        first_3_lines = [f.readline().strip() for _ in range(3)]
    
    d = pd.read_csv(
        file, header=None, decimal=".", sep=r"\s+", engine="python", skiprows=3).to_numpy(dtype=float)
    
    key = file.stem[2:] if not file.stem.startswith('Hg') else file.stem
    data[key] = {
        't': np.array(d[:, 0]), 
        'A': np.array(d[:, 1]),
        'header': first_3_lines
    }
data[key]['A'] = -1 * (data[key]['A'] + np.min(-data[key]['A']))

peaks_hg, props_hg = find_peaks(data[key]['A'], prominence=0.03, height=0.012)

widths_hg = peak_widths(data[key]['A'], peaks_hg, rel_height=0.2)

s_hg = DatasetSpec(data[key]['t'], data[key]['A'], label=key, marker='None', line='-')
s_peaks = DatasetSpec(data[key]['t'][peaks_hg], data[key]['A'][peaks_hg], label='Peaks', marker='x', line='None')



line_widths = np.array([])
line_widths_heights = np.array([])
results_Hg_y = np.array([])
t_gauss = np.array([])


for i in range(len(peaks_hg)):
    
    prominences = props_hg['prominences'][i]
    widths = widths_hg[0][i] *0.1
    width_heights = widths_hg[1][i]
    left_ips = widths_hg[2][i]
    right_ips = widths_hg[3][i]
    
    line_widths = np.append(line_widths, data[key]['t'][int(left_ips):int(right_ips)+1], axis=0)
    line_widths_heights = np.append(line_widths_heights, width_heights * np.ones(int(right_ips) - int(left_ips) + 1), axis=0)
    
    
    mask = (data[key]['t'] >= data[key]['t'][int(left_ips) - int(1*widths/2)]) & (data[key]['t'] <= data[key]['t'][int(right_ips) + int(1*widths/2)])
    t_per_peak = data[key]['t'][mask]
    A_per_peak = data[key]['A'][mask]
    
    # Add NaNs to line_widths_heights to separate peaks
    line_widths_heights = np.append(line_widths_heights, [np.nan], axis=0)
    line_widths = np.append(line_widths, [line_widths[-1]+0.1], axis=0)
    
    times_gauss = np.linspace(t_per_peak[0], t_per_peak[-1], 100)
    
    result_Hg = lmfit(
        xdata= t_per_peak,
        ydata= A_per_peak,
        model="gaussian",
        initial_params={'c': widths/2.355, 'b': data[key]['t'][peaks_hg[i]]} # a * np.exp(-((x - b) / c)**2 / 2) + d
    )
    
    results_Hg_y = np.append(results_Hg_y, result_Hg.eval(x=times_gauss), axis=0)
    t_gauss = np.append(t_gauss, times_gauss, axis=0)
    
    
    centering = (result_Hg.params['b'].value)[5]
    centering_wavelength = 507.3034
    print(f"Fitted peak at {data[key]['t'][peaks_hg[i]]:.4g} s:, Wavelength={(centering_wavelength-350)/centering*data[key]['t'][peaks_hg[i]] + 350}, Peak={100/4.172*result_Hg.eval(x=result_Hg.params['b'].value):.4g}, Center={result_Hg.params['b'].value:.4g}s, Prominence={prominences:.4g}")

    
s_widths = DatasetSpec(
    line_widths,
    line_widths_heights,
    label=f'Width', marker='None', line='--'
)

s_gauss = DatasetSpec(
    t_gauss,
    results_Hg_y,
    label=f'Gaussian Fit', marker='.', line='None', linewidth=2
)


plot_data(
    [s_hg, s_peaks, s_widths, s_gauss],
    title=f'Ruby Data: Hg',
    xlabel='Time (s)',
    ylabel='Amplitude',
    filename=f'Plots/Hg_lines.pdf',
    height=15,
    plot=False
)



for key in data.keys():
    if key == 'Hg_lines':
        continue
    plot_data(
        [DatasetSpec(data[key]['t'], data[key]['A'], label=key, marker='None', line='-')],
        title=f'Ruby Data: {key}',
        xlabel='Time (s)',
        ylabel='Amplitude',
        filename=f'Plots/{key}.pdf',
        height=15,
        plot=False
    )


Hg_lines = [253.6517*2, 365.0153, 398.3931, 404.6563, 435.8328, 546.0735, 567.7105, 579.663, 614.9475]

