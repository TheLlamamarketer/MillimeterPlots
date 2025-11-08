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

source_dir = Path('FP/HOL/Data/S 15.6 Alte Daten')
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
    if file.name.lower().startswith('de'):
        continue
    arr = pd.read_csv(
        file, header=None, decimal=",", sep=r"\s+", engine="python"
    ).iloc[:, 0].to_numpy(dtype=float)
    key = safe_name(file.stem)
    data_by_name[key] = arr[1:]

t_by_name = {k: np.arange(v.size) * t_step for k, v in data_by_name.items()}

cutoff_indexes = {
    'z_0cm': slice(59000, 71000),
    'z_1cm': slice(54800, 70000),
    'z_0_4cm': slice(30000, 41750),
    'z_0_5cm': slice(32220, 43000),
}

dz = np.array([0, -1, -0.4, 0.5]) * 2


# --- filter out any malformed names (e.g. from bad filenames that produced 'inf') ---
bad_keys = [k for k in list(data_by_name.keys()) if ('inf' in k) or (k.strip() == '' )]
for k in bad_keys:
    data_by_name.pop(k, None)


arr = np.array([data_by_name[k][:3500] for k in data_by_name.keys()]) if data_by_name else np.empty((0,))
if arr.size:
    baseline = arr.mean(axis=1)
    dbaseline = arr.std(axis=1, ddof=1) / np.sqrt(arr.shape[1])
else:
    baseline = np.zeros(0)
    dbaseline = np.zeros(0)

print("Baseline values and errors for old data:")
for b, db in zip(baseline, dbaseline):
    print(f"  {b:.3f} ± {db:.2e}")


s = []
K_all = np.array([])
dK_all = np.array([])

U_max = np.array([])
U_min = np.array([])

dU_max = np.array([])
dU_min = np.array([])

def weighted_error(values, weights):
    wsum = np.sum(weights)
    if wsum == 0:
        return np.nan
    mean = np.average(values, weights=weights)
    # weighted variance (with Bessel correction via effective N)
    var = np.sum(weights * (values - mean)**2) / (wsum - np.sum(weights**2)/wsum)
    se  = np.sqrt(var / len(values))
    return se

for k, v in data_by_name.items():
    i = list(data_by_name.keys()).index(k)
    # subtract baseline if available
    if i < len(baseline):
        v = v - baseline[i]

    seg  = v[cutoff_indexes.get(k, slice(None))]
    tseg = t_by_name[k][cutoff_indexes.get(k, slice(None))]

    s1 = DatasetSpec(x=tseg, y=seg, label= f"d={dz[i]}", marker='None', line='-', color_group=k)
    s.append(s1)

    # robust noise estimate
    mad    = np.median(np.abs(seg - np.median(seg))) if seg.size else 0
    sigma  = 1.4826*mad
    rng    = np.ptp(seg) if seg.size else 0

    # find peaks with adaptive prominence
    pks_loose, _ = find_peaks(seg,  prominence=sigma) if seg.size else (np.array([]), {})
    period = np.median(np.diff(pks_loose)) if pks_loose.size>1 else 1

    prom0  = max(2.5*sigma, 0.05*rng)
    min_distance = int(0.5 * period) if period>0 else 1

    prom = prom0
    prop_max = {}
    prop_min = {}
    pks_max = np.array([])
    pks_min = np.array([])
    for _ in range(3):
        pks_max, prop_max = find_peaks(seg,  prominence=prom, distance=min_distance)
        pks_min, prop_min = find_peaks(-seg, prominence=prom, distance=min_distance)
        if len(pks_max) >= 5 and len(pks_min) >= 5:
            break
        prom *= 0.7

    pk_vals = seg[pks_max] if pks_max.size else np.array([])
    tr_vals = seg[pks_min] if pks_min.size else np.array([])

    cut_hi = np.quantile(pk_vals, 0.90) if pk_vals.size>0 else np.nan
    cut_lo = np.quantile(tr_vals, 0.10) if tr_vals.size>0 else np.nan

    if pk_vals.size>0:
        peaks_max = pks_max[pk_vals >= cut_hi]
    else:
        peaks_max = pks_max
    if tr_vals.size>0:
        peaks_min = pks_min[tr_vals <= cut_lo]
    else:
        peaks_min = pks_min

    prom_max = np.asarray(prop_max.get('prominences', np.ones_like(pks_max)))
    prom_min = np.asarray(prop_min.get('prominences', np.ones_like(pks_min)))

    wgt_max = prom_max / (sigma**2 + 1e-12)
    wgt_min = prom_min / (sigma**2 + 1e-12)

    Umax = np.average(seg[pks_max], weights=wgt_max) if pks_max.size else np.nan
    Umin = np.average(seg[pks_min], weights=wgt_min) if pks_min.size else np.nan

    U_max = np.append(U_max, Umax)
    U_min = np.append(U_min, Umin)

    dUmax = weighted_error(seg[pks_max], wgt_max) if pks_max.size else np.nan
    dUmin = weighted_error(seg[pks_min], wgt_min) if pks_min.size else np.nan

    dU_max = np.append(dU_max, dUmax)
    dU_min = np.append(dU_min, dUmin)

    if not (np.isnan(Umax) or np.isnan(Umin)):
        dK = 2.0 * np.sqrt( (Umax*dUmin)**2 + (Umin*dUmax)**2 ) / (Umax + Umin)**2
        K = (Umax - Umin) / (Umax + Umin)
    else:
        dK = np.nan
        K = np.nan

    dK_all = np.append(dK_all, dK)
    K_all = np.append(K_all, K)

    t_max = tseg[pks_max] if pks_max.size else np.array([])
    y_max = seg[pks_max] if pks_max.size else np.array([])
    t_min = tseg[pks_min] if pks_min.size else np.array([])
    y_min = seg[pks_min] if pks_min.size else np.array([])

    smax = DatasetSpec(x=t_max, y=y_max, fit_x= np.linspace(tseg.min(), tseg.max(), 5) if seg.size else np.array([]), fit_y=[Umax]*5 if seg.size else np.array([]), label= f"Peaks d={k}", marker='x', line='None', color='red', fit_line='--', fit_label=f"Avg. Peak: {Umax:.2f}")
    smin = DatasetSpec(x=t_min, y=y_min, fit_x= np.linspace(tseg.min(), tseg.max(), 5) if seg.size else np.array([]), fit_y=[Umin]*5 if seg.size else np.array([]), label= f"Minima d={k}", marker='x', line='None', color='blue', fit_line='--', fit_label=f"Avg. Min: {Umin:.2f}")

    plot_data(
        filename=f"Plots/Hol_old_{k}.pdf",
        datasets=[s1, smax, smin],
        xlabel="Time t/s ",
        ylabel="Intensity U/V",
        title="Michelson Interferometer (old)",
        color_seed=72,
        plot=True
    )

    data_by_name[k] = v


# Prepare fits for K_old computed from U_max/U_min per-file
x_fits = np.linspace(dz[0], dz[-1], 100)

eps = np.finfo(float).eps
lower_ok = np.maximum(K_all - dK_all, eps)
y_old = np.log(K_all)
yerr_lower_old = y_old - np.log(lower_ok)
yerr_upper_old = np.log(K_all + dK_all) - y_old

# perform fit (use lmfit from Functions.help)
results_K_old = lmfit(dz, y_old, yerr=(0.5*(yerr_lower_old + yerr_upper_old)), model="linear")

print(results_K_old.fit_report())

s_K_old = DatasetSpec(
    x=np.array(dz),
    y=y_old,
    fit_x=x_fits,
    fit_y=results_K_old.eval(x=x_fits),
    confidence=calc_CI(results_K_old, x_fits, sigmas=[1]),
    xerr=0.2,
    yerr=(yerr_lower_old, yerr_upper_old),   # asymmetric log errors
    label="K_old(d)",
    marker='.', line='None', color_group='old'
)

plot_data(
    filename=f"Plots/Hol_K_old.pdf",
    datasets=[s_K_old],
    xlabel="z/cm ",
    ylabel="log(K)",
    title="Michelson Interferometer - log(Kontrast) (old)",
    color_seed=72,
    plot=True
)

