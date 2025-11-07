import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

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

# read every file in the folder 
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


print(data_by_name.values())

arr = np.array([data_by_name[k][:3500] for k in data_by_name.keys()])  
baseline = arr.mean(axis=1)                                               
dbaseline = arr.std(axis=1, ddof=1) / np.sqrt(arr.shape[1])

s = []
max_min = np.array([])
max_min_errors = np.array([])

for k, v in data_by_name.items():
    i = list(data_by_name.keys()).index(k)
    v = v - baseline[i]
    
    seg  = v[cutoff_indexes[k]]
    tseg = t_by_name[k][cutoff_indexes[k]]

    s1 = DatasetSpec(x=tseg, y=seg, label= f"d={dz[i]}", marker='None', line='-', color_group=k)
    s.append(s1)
    
    

    # --- robust noise + range
    mad    = np.median(np.abs(seg - np.median(seg)))
    sigma  = 1.4826*mad
    rng    = np.ptp(seg) + 1e-12

    # Base prominence: combine absolute (σ) and relative (range)
    prom0  = max(2.5*sigma, 0.05*rng)     # 2.5–4 × σ and ≥5% of local range
    dist0  = max(1, len(seg)//200)        # gentle spacing constraint

    def pick_extrema(prom):
        pksM, propM = find_peaks(seg,  prominence=prom, distance=dist0)
        pksm, propm = find_peaks(-seg, prominence=prom, distance=dist0)
        return pksM, propM, pksm, propm

    # --- auto-relax if we got too few
    prom = prom0
    for _ in range(3):
        pks_max, prop_max, pks_min, prop_min = pick_extrema(prom)
        if len(pks_max) >= 3 and len(pks_min) >= 3:
            break
        prom *= 0.7  # relax by 30% and try again

    # If still empty, fall back to top/bottom quantiles to avoid NaNs
    if len(pks_max) == 0:
        thr = np.quantile(seg, 0.90)
        pks_max = np.where(seg >= thr)[0]
        prop_max = {'prominences': np.full_like(pks_max, thr - np.median(seg), dtype=float)}
    if len(pks_min) == 0:
        thr = np.quantile(seg, 0.10)
        pks_min = np.where(seg <= thr)[0]
        prop_min = {'prominences': np.full_like(pks_min, np.median(seg) - thr, dtype=float)}

    # --- widths and quality weights
    w_max = peak_widths(seg,  pks_max, rel_height=0.5)[0] if len(pks_max) else np.array([])
    w_min = peak_widths(-seg, pks_min, rel_height=0.5)[0] if len(pks_min) else np.array([])

    prom_max = np.asarray(prop_max.get('prominences', np.ones_like(pks_max)))
    prom_min = np.asarray(prop_min.get('prominences', np.ones_like(pks_min)))

    wgt_max = (prom_max * (w_max + 1e-12)) / ((sigma**2) + 1e-12)
    wgt_min = (prom_min * (w_min + 1e-12)) / ((sigma**2) + 1e-12)

    # Cap extreme weights (robustness)
    def cap(w):
        if w.size == 0: return w
        capv = np.quantile(w, 0.95)
        return np.minimum(w, capv)
    wgt_max = cap(wgt_max); wgt_min = cap(wgt_min)

    def wavg(y, w):
        if y.size == 0: return (np.nan, np.nan)
        mu = np.average(y, weights=w)
        # effective sample size
        neff = (w.sum()**2) / (np.sum(w**2) + 1e-12)
        var = np.average((y - mu)**2, weights=w) / max(neff, 1.0)
        return mu, np.sqrt(max(var, 0.0))

    peaks_max_avg, err_max = wavg(seg[pks_max], wgt_max)
    peaks_min_avg, err_min = wavg(seg[pks_min], wgt_min)

    # bookkeeping
    max_min = np.append(max_min, [peaks_max_avg, peaks_min_avg], axis=0)
    max_min_errors = np.append(max_min_errors, [err_max, err_min], axis=0)
    t_max = tseg[pks_max]; y_max = seg[pks_max]
    t_min = tseg[pks_min]; y_min = seg[pks_min]
    
    """
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
    
    """

    smax = DatasetSpec(x=t_max, y=y_max, fit_x= np.linspace(tseg.min(), tseg.max(), 5), fit_y=[peaks_max_avg]*5, label= f"Peaks dz={k}", marker='x', line='None', color='red', fit_line='--', fit_label=f"Avg. Peak: {peaks_max_avg:.2f}")
    smin = DatasetSpec(x=t_min, y=y_min, fit_x= np.linspace(tseg.min(), tseg.max(), 5), fit_y=[peaks_min_avg]*5, label= f"Minima dz={k}", marker='x', line='None', color='blue', fit_line='--', fit_label=f"Avg. Min: {peaks_min_avg:.2f}")


    plot_data(
        filename=f"Plots/Hol_{k}.pdf",
        datasets=[s1, smax, smin],
        xlabel="Time t/s ",
        ylabel="Intensität I",
        title="Michelson Interferometer",
        color_seed=72,
        plot=True
    )
    
    
    data_by_name[k] = v

x_fits = np.linspace(dz[0], dz[-1], 100)

A = max_min[::2]          
B = max_min[1::2]         
sA = max_min_errors[::2]
sB = max_min_errors[1::2]

K = (A - B) / (A + B)

dK = 2.0 * np.sqrt( (B**2) * (sA**2) + (A**2) * (sB**2) ) / (A + B)**2  

log_sigma = dK / K
results_K = lmfit(dz, np.log(K), yerr=log_sigma, model="linear")

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


background = np.array([
    0.088028, 0.091549, 0.094366, 0.098591,
    0.095774, 0.0908419, 0.095774, 0.094366,
    0.098591, 0.098591, 0.097183, 0.092957
])

I_max = np.array([
    1.98, 1.96, 2.09, 1.79,
    2.07, 1.92, 1.91, 1.91,
    1.86, 1.82, 1.82, 1.63
])

I_min = np.array([
    0.14, 0.25, 0.07, 0.29,
    0.10, 0.28, 0.26, 0.35,
    0.40, 0.38, 0.38, 0.56
])

I_max_err = np.array([
    0.07, 0.12, 0.06, 0.07,
    0.04, 0.05, 0.10, 0.11,
    0.07, 0.07, 0.07, 0.06
])

I_min_err = np.array([
    0.02, 0.02, 0.03, 0.04,
    0.02, 0.07, 0.08, 0.11,
    0.08, 0.07, 0.07, 0.06
])


data_old = {
    'background': background,
    'd': dz,
    'I_max': I_max,
    'I_min': I_min,
    'I_max_err': I_max_err,
    'I_min_err': I_min_err
}

headers_old = {
    "d":          {"label": "{$d_{ges}$/cm}", "intermed": True, "err": 0.2},
    "background": {"label": "{$hintergrund$}", "round": False},
    "I_max":      {"label": "{$U_{max}$}", "intermed": True, "err": I_max_err},
    "I_min":      {"label": "{$U_{min}$}", "intermed": True, "err": I_min_err}
}


print_standard_table(data=data_old,
    headers=headers_old,
    header_groups=[("", 1), ("Messwerte U/V", 3)],
    caption="Alte Messdaten des Michelson-Interferometers.",
    label="tab:hol_alte_daten",
    column_formats=["S[table-format=1.2]", "S[table-format=1.5]", "S[table-format=1.2]", "S[table-format=1.2]"],
    si_setup=None,
    show=False,
)

K_old = (I_max - I_min) / (I_max + I_min - 2 * background)

dK_old = 2.0 * np.sqrt( (B**2) * (sA**2) + (A**2) * (sB**2) ) / (A + B)**2  

log_sigma = dK_old / K_old
results_K_old = lmfit(dz, np.log(K_old), yerr=log_sigma, model="linear")


y_old = np.log(K_old)
yerr_lower_old = y_old - np.log(K_old - dK_old)
yerr_upper_old = np.log(K_old + dK_old) - y_old

print(results_K_old.fit_report())

s_K_old = DatasetSpec(
    x=np.array(dz),
    y=y_old,
    fit_x=x_fits,
    fit_y=results_K_old.eval(x=x_fits),
    confidence=calc_CI(results_K_old, x_fits, sigmas=[1]),
    xerr=0.2,
    yerr=(yerr_lower_old, yerr_upper_old),   # correct asymmetric log errors
    label="K_old(z)",
    marker='.', line='None', color_group='old'
)


plot_data(
    filename=f"Plots/Hol_K.pdf",
    datasets=[s_K, s_K_old],
    xlabel="z/cm ",
    ylabel="log(K)",
    title="Michelson Interferometer - log(Kontrast)",
    color_seed=72,
    plot=True
)