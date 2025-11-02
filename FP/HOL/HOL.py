import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from Functions.plotting import DatasetSpec, plot_data
from Functions.tables import *
from Functions.help import *

source_dir = Path('FP/HOL/Data/S 8.8 Neue Daten')
t_step = 0.001  

def safe_name(stem: str) -> str:
    """Turn a filename stem into a valid Python identifier."""
    name = re.sub(r'\W+', '_', stem)      # replace non-alnum with _
    name = re.sub(r'_Interferenz_z_', '', name[7:])
    if name and name[0].isdigit():
        name = f'z_{name}'               # identifiers can't start with a digit
    return name or 'data'

data_by_name: dict[str, np.ndarray] = {}

# read every file in the folder (adjust the glob if you need a specific extension)
for file in sorted(source_dir.iterdir()):
    if not file.is_file():
        continue
    arr = pd.read_csv(
        file, header=None, decimal=",", sep=r"\s+", engine="python"
    ).iloc[:, 0].to_numpy(dtype=float)
    key = safe_name(file.stem)
    data_by_name[key] = arr[1:]

t_by_name = {k: np.arange(v.size) * t_step for k, v in data_by_name.items()}

cutoff_indexes = {
    'z_0_8cm': slice(14480, 21900),  
    'z_0_3cm': slice(26185, 32930),  
    'z_0cm':   slice(13840, 25500),  
    'z_0_2cm': slice(18195, 28650),
    'z_0_7cm': slice(12275, 24300),
    'z_1_2cm': slice(18130, 26500),
    'z_1_7cm': slice(16630, 23350),
    'z_2_2cm': slice(11120, 19550),
    'z_3_2cm': slice(13680, 21880),
    'z_4_2cm': slice(8695, 14200),
    'z_5_2cm': slice(7970, 15240),
    'z_6_2cm': slice(5940, 15300),
}

dz = np.array([-0.8, -0.3, 0.0, 0.2, 0.7, 1.2, 1.7, 2.2, 3.2, 4.2, 5.2, 6.2]) * 2
baseline = np.mean(np.concatenate([v[:3500] for v in data_by_name.values()]))


s = []
max_min = np.array([])
max_min_errors = np.array([])


for k in list(data_by_name.keys()):
    data_by_name[k] = data_by_name[k] - baseline

for k, v in data_by_name.items():
    s1 = DatasetSpec(x=t_by_name[k][cutoff_indexes[k]], y=v[cutoff_indexes[k]], label= f"dz={k}", marker='None', line='-', color_group=k)
    s.append(s1)
    
    max_avg = np.mean(v[cutoff_indexes[k]][find_peaks(v[cutoff_indexes[k]], prominence=0.9)[0]])
    min_avg = np.mean(v[cutoff_indexes[k]][find_peaks(-v[cutoff_indexes[k]], prominence=0.9)[0]])
    
    mean = (max_avg + min_avg) / 2
    
    print(f"dz={k}: max_avg={max_avg}, min_avg={min_avg}, mean={mean}")

    peaks_max = find_peaks(v[cutoff_indexes[k]], prominence=0.9, height=mean + (max_avg - mean) * 0.9)[0]
    peaks_min = find_peaks(-v[cutoff_indexes[k]], prominence=0.9, height=-mean - (min_avg - mean) * 0.9)[0]

    peaks_max_avg = np.average(v[cutoff_indexes[k]][peaks_max], weights=v[cutoff_indexes[k]][peaks_max]/max_avg*10)
    peaks_min_avg = np.average(v[cutoff_indexes[k]][peaks_min], weights=v[cutoff_indexes[k]][peaks_min]/min_avg*10)

    max_min = np.append(max_min, [peaks_max_avg, peaks_min_avg], axis=0)
    max_min_errors = np.append(max_min_errors, [np.std(v[cutoff_indexes[k]][peaks_max]), np.std(v[cutoff_indexes[k]][peaks_min])], axis=0)

    t_max = t_by_name[k][cutoff_indexes[k]][peaks_max]
    y_max = v[cutoff_indexes[k]][peaks_max]
    t_min = t_by_name[k][cutoff_indexes[k]][peaks_min]
    y_min = v[cutoff_indexes[k]][peaks_min]


    smax = DatasetSpec(x=t_max, y=y_max, fit_x= np.linspace(t_by_name[k][cutoff_indexes[k]].min(), t_by_name[k][cutoff_indexes[k]].max(), 5), fit_y=[peaks_max_avg]*5, label= f"Peaks dz={k}", marker='x', line='None', color='red', fit_line='--', fit_label=f"Avg. Peak: {peaks_max_avg:.2f}")
    smin = DatasetSpec(x=t_min, y=y_min, fit_x= np.linspace(t_by_name[k][cutoff_indexes[k]].min(), t_by_name[k][cutoff_indexes[k]].max(), 5), fit_y=[peaks_min_avg]*5, label= f"Minima dz={k}", marker='x', line='None', color='blue', fit_line='--', fit_label=f"Avg. Min: {peaks_min_avg:.2f}")

    plot_data(
        filename=f"Plots/Hol_{k}.pdf",
        datasets=[s1, smax, smin],
        xlabel="Time t/s ",
        ylabel="Intensität I",
        title="Michelson Interferometer",
        color_seed=54,
        plot=False
    )

x_fits = np.linspace(dz[0], dz[-1], 100)

A = max_min[::2]          
B = max_min[1::2]         
sA = max_min_errors[::2]
sB = max_min_errors[1::2]

K = (A - B) / (A + B)

dK = 2.0 * np.sqrt( (B**2) * (sA**2) + (A**2) * (sB**2) ) / (A + B)**2  

log_sigma = dK / K
results_K = lmfit(np.array(dz), np.log(K), yerr=log_sigma, model="linear")

print(results_K.fit_report())


eps = np.finfo(float).eps
lower_ok = np.maximum(K - dK, eps)    
y = np.log(K)
yerr_lower = y - np.log(lower_ok)
yerr_upper = np.log(K + dK) - y

s_K = DatasetSpec(
    x=np.array(dz),
    y=y,
    fit_x=x_fits,
    fit_y=results_K.eval(x=x_fits),
    confidence=calc_CI(results_K, x_fits, sigmas=[1]),
    xerr=0.2,
    yerr=(yerr_lower, yerr_upper),   # correct asymmetric log errors
    label="K(z)",
    marker='.', line='None'
)

plot_data(
    filename=f"Plots/Hol_K.pdf",
    datasets=[s_K],
    xlabel="z/cm ",
    ylabel="log(K)",
    title="Michelson Interferometer - log(Kontrast)",
    color_seed=72,
    plot=True
)

