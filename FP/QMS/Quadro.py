import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, peak_widths, peak_prominences

from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).parent.parent.parent))

from Functions.plotting import *
from Functions.tables import *
from Functions.help import *
from scipy.interpolate import UnivariateSpline
from collections import defaultdict


source_dir = Path(__file__).resolve().parent / "Data"

massrs, signals = [], []
names, pressures = [], []
data = []

for f in source_dir.glob('*.txt'):
    p = f.stem.find('_p1')
    redo = f.stem.find('_redo')
    name = f.stem[:p] + f.stem[p+5:]
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

sU, s_peaks = [], []
for i, (name, d, pressure) in enumerate(zip(voltage_run['name'][:], voltage_run['data'][:], voltage_run['pressure'][:])):
    mr, s = d
    b0 = np.quantile(s, 0.1)
    s -= b0
    color = plt.cm.plasma(i / len(voltage_run['name'][:])) # all colormaps are: plasma, viridis, inferno, magma, cividis
    sU.append(DatasetSpec(x=mr, y=s, label=f'$U_B = {U_B[i]:.2f} V$', line='-', marker='None', color=color, linewidth=0.5))
    
    peaks, props = find_peaks(s, prominence=5, distance=30, height=2.0)
    
    if peaks.size > 2:
        top2_idx = np.argsort(s[peaks])[-2:]
        peaks = peaks[top2_idx]
        props = {key: val[top2_idx] for key, val in props.items()}

    # FWHM
    widths_idx, width_heights, left_ips, right_ips = peak_widths(s, peaks, rel_height=0.5)
    idx = np.arange(s.size)
    m_left  = np.interp(left_ips,  idx, mr)
    m_right = np.interp(right_ips, idx, mr)

    mr_peaks = mr[peaks]
    signal_peaks = s[peaks]
    delta = m_right - m_left
    res = mr_peaks / delta
    
    print(f"Run: {name}, Peaks at m/z: {mr_peaks}, FWHM: {delta}, Resolution: {res}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mr, s, label="Signal")
    ax.plot(mr_peaks, signal_peaks, "x", markersize=12, label="Peaks")

    # FWHM graphics
    for k in range(peaks.size):
        ax.hlines(width_heights[k], m_left[k], m_right[k], linewidth=2, label="FWHM" if k == 0 else None)
        ax.vlines([m_left[k], m_right[k]], ymin=width_heights[k] - 0.5, ymax=width_heights[k] + 0.5,
                linestyles="dotted", linewidth=1, label="Width edges" if k == 0 else None)
    
    if "prominences" in props:
        prom = props["prominences"]
        lb = props["left_bases"]
        rb = props["right_bases"]

        # contour line height used for prominence
        y0 = s[peaks] - prom

        for k, p in enumerate(peaks):
            ax.vlines(mr[p], y0[k], s[p], lw=2, alpha=0.7, label="Prominence" if k == 0 else None)
    ax.set_xlabel(r"Massenladungszahl ($m/z$)")
    ax.set_ylabel("Signalstärke (arb. Einheiten)")
    ax.legend()
    plt.close()

    
    s_peaks.append(DatasetSpec(x=mr_peaks, y=signal_peaks, line='None', marker='x', color='red', markersize=12))


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




sorting_res = np.argsort([int(name[5]) for name in resolution_run['name']])
resolution_run['name'] = [resolution_run['name'][i] for i in sorting_res]
resolution_run['data'] = [resolution_run['data'][i] for i in sorting_res]
resolution_run['pressure'] = [resolution_run['pressure'][i] for i in sorting_res]


s_res, s_peaks = [], []
for name, d, pressure in zip(resolution_run['name'], resolution_run['data'], resolution_run['pressure']):
    mr, s = d
    b0 = np.quantile(s, 0.1)
    s -= b0
    
    peaks, props = find_peaks(s, prominence=0.8*np.sqrt(max(s)), distance=30)
    mr_peaks = mr[peaks]
    signal_peaks = s[peaks]
    
    width_samples, height_eval, left_ips, right_ips = peak_widths(s, peaks, rel_height=0.5)
    
    color = plt.cm.viridis(int(name[5])*10 / 100) # all colormaps are: plasma, viridis, inferno, magma, cividis
    s_res.append(DatasetSpec(x=mr, y=s, label=f'$Res = {name[5]}$', line='-', marker='None', color=color, linewidth=0.5))
    s_peaks.append(DatasetSpec(x=mr_peaks, y=signal_peaks, line='None', marker='x', color='red', markersize=12))

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
    
    b0 = np.quantile(s, 0.1)
    s -= b0
    mask2 = s > 0
    s = s[mask2]
    mr = mr[mask2]
    s_filtered = savgol_filter(s, 31, 3)
    
    s/= max(s_filtered) /100
    s_filtered /= max(s_filtered)/100
    
    
    
    peaks, _ = find_peaks(s_filtered, prominence=5, distance=30)
    mr_peaks = mr[peaks]
    s_filtered_peaks = s_filtered[peaks]
        
    s_theory = DatasetSpec(x=data[name_mol]['mz']-0.1, y=data[name_mol]['s'], label=f'${name_mol}$ Theorie', color='tab:purple', plot_style='bar', barwidth=0.2)
    s_mol= DatasetSpec(x=mr, y=s, label=f"${name_mol}$", line='-', marker='None',color='tab:green', fit_y=s_filtered, fit_x=mr, fit_label=f'${name_mol}$ Fit', fit_color='tab:blue', linewidth=0.5)
    s_peaks = DatasetSpec(x=mr_peaks, y=s_filtered_peaks, label=f'${name_mol}$ Peaks', line='None', marker='x', color='tab:red', markersize=24)
    s_peaks_bar = DatasetSpec(x=np.rint(mr_peaks).astype(int)+0.1, y=s_filtered_peaks, label=f'${name_mol}$ Peaks', plot_style='bar', barwidth=0.2)

    plot_data(
        [s_theory, s_peaks_bar, s_mol, s_peaks],
        filename=f'Plots/{name}.pdf',
        title=f'Signalstärke von {name.replace("_", " ")} im Vergleich zur Theorie',
        xlabel='Massenladungszahl ($m/z$)',
        ylabel='Signalstärke (arb. Einheiten)',
        width=20,
        height=10,
        color_seed=33,
        plot=False,
    )
    
    
    plot_data(
        [s_theory, s_peaks_bar],
        filename=f'Plots/{name}_spectrum.pdf',
        title=f'Massenspektrum von {name.replace("_", " ")} im Vergleich zur Theorie',
        xlabel='Massenladungszahl ($m/z$)',
        ylabel='Signalstärke (arb. Einheiten)',
        xlim=data[name_mol]['mask'],
        width=20,
        height=10,
        color_seed=33,
        plot=False,
    )

