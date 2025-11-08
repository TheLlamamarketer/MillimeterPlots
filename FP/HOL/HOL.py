import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

arr = np.array([data_by_name[k][:3500] for k in data_by_name.keys()])  
baseline = arr.mean(axis=1)                                               
dbaseline = arr.std(axis=1, ddof=1) / np.sqrt(arr.shape[1])

print("Baseline values and errors:")
for b, db in zip(baseline, dbaseline):
    print(f"  {b:.3f} ± {db:.2e}")

s = []
K_all = np.array([])
dK_all = np.array([])

U_max = np.array([])
U_min = np.array([])

dU_max = np.array([])
dU_min = np.array([])

for k, v in data_by_name.items():
    i = list(data_by_name.keys()).index(k)
    v = v - baseline[i]
    
    seg  = v[cutoff_indexes[k]]
    tseg = t_by_name[k][cutoff_indexes[k]]

    s1 = DatasetSpec(x=tseg, y=seg, label= f"d={dz[i]}", marker='None', line='-', color_group=k)
    s.append(s1)

    mad    = np.median(np.abs(seg - np.median(seg))) # Median Absolute Deviation as a estimator of dispersion
    sigma  = 1.4826*mad
    rng    = np.ptp(seg)                             # estimate of amplitude range
    
    pks_loose, _ = find_peaks(seg, prominence=2*sigma)
    period = np.median(np.diff(pks_loose))

    prom0  = max(2.5*sigma, 0.05*rng)
    min_distance = int(0.5 * period)

    prom = prom0
    for _ in range(3):
        pks_max, prop_max = find_peaks(seg,  prominence=prom, distance=min_distance)
        pks_min, prop_min = find_peaks(-seg, prominence=prom, distance=min_distance)
        if len(pks_max) >= 30 and len(pks_min) >= 30:
            break
        prom *= 0.7  # relax by 30% and try again

    pk_vals = seg[pks_max]
    tr_vals = seg[pks_min]
    
    cut_hi = np.quantile(pk_vals, 0.90) if len(pk_vals) > 0 else np.nan
    cut_lo = np.quantile(tr_vals, 0.10) if len(tr_vals) > 0 else np.nan
    
    peaks_max = pks_max[pk_vals >= cut_hi]
    peaks_min = pks_min[tr_vals <= cut_lo]

    prom_max = np.asarray(prop_max.get('prominences', np.ones_like(pks_max)))
    prom_min = np.asarray(prop_min.get('prominences', np.ones_like(pks_min)))

    wgt_max = prom_max / (sigma**2 + 1e-12)
    wgt_min = prom_min / (sigma**2 + 1e-12)
    
    Umax = np.average(seg[pks_max], weights=wgt_max)
    Umin = np.average(seg[pks_min], weights=wgt_min)
    
    U_max = np.append(U_max, Umax)
    U_min = np.append(U_min, Umin)

    K = (Umax - Umin) / (Umax + Umin)
    K_all = np.append(K_all, K)
    
    def weighted_error(values, weights):
        wsum = np.sum(weights)
        if wsum == 0:
            return np.nan
        mean = np.average(values, weights=weights)
        # weighted variance (with Bessel correction via effective N)
        var = np.sum(weights * (values - mean)**2) / (wsum - np.sum(weights**2)/wsum)
        se  = np.sqrt(var / len(values))
        return se

    dUmax = weighted_error(seg[pks_max], wgt_max)
    dUmin = weighted_error(seg[pks_min], wgt_min)
    
    dU_max = np.append(dU_max, dUmax)
    dU_min = np.append(dU_min, dUmin)

    dK = 2.0 * np.sqrt( (Umax*dUmin)**2 + (Umin*dUmax)**2 ) / (Umax + Umin)**2
    dK_all = np.append(dK_all, dK)

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

    smax = DatasetSpec(x=t_max, y=y_max, fit_x= np.linspace(tseg.min(), tseg.max(), 5), fit_y=[Umax]*5, label= f"Peaks d={k}", marker='x', line='None', color='red', fit_line='--', fit_label=f"Avg. Peak: {Umax:.2f}")
    smin = DatasetSpec(x=t_min, y=y_min, fit_x= np.linspace(tseg.min(), tseg.max(), 5), fit_y=[Umin]*5, label= f"Minima d={k}", marker='x', line='None', color='blue', fit_line='--', fit_label=f"Avg. Min: {Umin:.2f}")

    plot_data(
        filename=f"Plots/Hol_{k}.pdf",
        datasets=[s1, smax, smin],
        xlabel="Time t/s ",
        ylabel="Intensity in U/V",
        title="Michelson Interferometer",
        color_seed=72,
        plot=False
    )
    
    data_by_name[k] = v

x_fits = np.linspace(dz[0], dz[-1], 100)


log_sigma = dK_all / K_all
results_K = lmfit(dz, np.log(K_all), model="linear")

print(results_K.fit_report())


eps = np.finfo(float).eps
lower_ok = np.maximum(K_all - dK_all, eps)
y = np.log(K_all)
yerr_lower = y - np.log(lower_ok)
yerr_upper = np.log(K_all + dK_all) - y

s_K = DatasetSpec(
    x=np.array(dz),
    y=y,
    fit_x=x_fits,
    fit_y=results_K.eval(x=x_fits),
    confidence=calc_CI(results_K, x_fits, sigmas=[1]),
    xerr=0.2,
    yerr=(yerr_lower, yerr_upper),   # correct asymmetric log errors
    label="K(d)",
    marker='.', line='None'
)

data_new = {
    'd': dz,
    'U_max': U_max,
    'U_min': U_min,
    'U_max_err': dU_max,
    'U_min_err': dU_min,
    'K_old': K_all,
    'dK_old': dK_all
}

headers_new = {
    "d":          {"label": "{$d$/cm}", "intermed": True, "err": 0.2},
    "U_max":      {"label": "{$U_{max}$}", "intermed": True, "err": dU_max},
    "U_min":      {"label": "{$U_{min}$}", "intermed": True, "err": dU_min},
    "K_old":      {"label": "{$K$}", "intermed": True, "err": dK_all}
}

print_standard_table(data=data_new,
    headers=headers_new,
    header_groups=[("d=2z", 1), ("Messwerte U/V ohne Nulltrate", 2), ("", 1)],
    caption="Neue Messdaten des Michelson-Interferometers.",
    label="tab:hol_neue_daten",
    column_formats=["S[table-format=1.2]", "S[table-format=1.2]", "S[table-format=1.2]", "S[table-format=1.2]"],
    si_setup=None,
    show=False,
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







K_old = (I_max - I_min) / (I_max + I_min - 2 * background)

dK_old = 2.0 * np.sqrt( (I_max * I_min_err)**2 + (I_min * I_max_err)**2 ) / (I_max + I_min)**2

results_K_old = lmfit(dz, np.log(K_old), model="linear")

data_old = {
    'background': background,
    'd': dz,
    'I_max': I_max,
    'I_min': I_min,
    'I_max_err': I_max_err,
    'I_min_err': I_min_err,
    'K_old': K_old,
    'dK_old': dK_old
}

y_old = np.log(K_old)
yerr_lower_old = y_old - np.log(K_old - dK_old)
yerr_upper_old = np.log(K_old + dK_old) - y_old

headers_old = {
    "d":          {"label": "{$d$/cm}", "intermed": True, "err": 0.2},
    "background": {"label": "{$Nulltrate$}", "intermed": True},
    "I_max":      {"label": "{$U_{max}$}", "intermed": True, "err": I_max_err},
    "I_min":      {"label": "{$U_{min}$}", "intermed": True, "err": I_min_err},
    "K_old":      {"label": "{$K$}", "intermed": True, "err": dK_old}
}




print_standard_table(data=data_old,
    headers=headers_old,
    header_groups=[("d=2z", 1), ("Messwerte U/V", 3), ("", 1)],
    caption="Alte Messdaten des Michelson-Interferometers.",
    label="tab:hol_alte_daten",
    column_formats=["S[table-format=1.2]", "S[table-format=1.5]", "S[table-format=1.2]", "S[table-format=1.2]", "S[table-format=1.2]"],
    si_setup=None,
    show=False,
)



print(results_K_old.fit_report())

s_K_old = DatasetSpec(
    x=np.array(dz),
    y=y_old,
    fit_x=x_fits,
    fit_y=results_K_old.eval(x=x_fits),
    confidence=calc_CI(results_K_old, x_fits, sigmas=[1]),
    xerr=0.2,
    yerr=(yerr_lower_old, yerr_upper_old),   # correct asymmetric log errors
    label="K(d) Programm",
    marker='.', line='None', color_group='old'
)

s_K_normal = DatasetSpec( x=np.array(dz), y=K_all, yerr=dK_all, xerr = 0.2, label="K(d)", marker='.', line='None' )
s_K_old_normal = DatasetSpec( x=np.array(dz), y=K_old, yerr=dK_old, xerr = 0.2, label="K(d) Programm", marker='.', line='None' )

plot_data(
    filename=f"Plots/Hol_K_normal.pdf",
    datasets=[s_K_normal, s_K_old_normal],
    xlabel="d/cm ",
    ylabel="K",
    title="Michelson Interferometer Kontrast K(d)",
    color_seed=72,
    plot=False
)



plot_data(
    filename=f"Plots/Hol_log_K.pdf",
    datasets=[s_K, s_K_old],
    xlabel="d/cm ",
    ylabel="$ln(K)$",
    title="Michelson Interferometer - ln(Kontrast)",
    color_seed=72,
    plot=True
)


print_standard_table(
    data={
        'a': [results_K.params['a'].value, results_K_old.params['a'].value],
        'b': [results_K.params['b'].value, results_K_old.params['b'].value],
        'R2': [results_K.rsquared, results_K_old.rsquared],
        'Parameters': ['{Python}', '{Programm}']
    },
    headers={
        "Parameters": {"label": "{}"},
        "a":          {"label": "{$a$}", "intermed": True, "err": results_K.params['a'].stderr},
        "b":   {"label": "{$b$}", "intermed": True, "err": results_K.params['b'].stderr},
        "R2":   {"label": "{$R^2$}", "intermed": True}
    },
    caption="Fit-Parameter für die Auswertung des Michelson-Interferometers.",
    label="tab:fit_parameter",
    column_formats=["S[table-format=1.2]", "S[table-format=1.2]", "S[table-format=1.2]", "S[table-format=1.2]"],
    si_setup=None,
    show=True,
)




d_coh_new = (-1 - results_K.params['a'].value) / results_K.params['b'].value
d_coh_old = (-1 - results_K_old.params['a'].value) / results_K_old.params['b'].value

dd_coh_new = np.sqrt(
    (results_K.params['a'].stderr / results_K.params['b'].value)**2 +
    ((1 + results_K.params['a'].value) * results_K.params['b'].stderr / results_K.params['b'].value**2)**2
)

dd_coh_old = np.sqrt(
    (results_K_old.params['a'].stderr / results_K_old.params['b'].value)**2 +
    ((1 + results_K_old.params['a'].value) * results_K_old.params['b'].stderr / results_K_old.params['b'].value**2)**2
)


print(f"Coherence length new data: d_coh = {d_coh_new:.1f} ± {dd_coh_new:.1f} cm")
print(f"Coherence length old data: d_coh = {d_coh_old:.1f} ± {dd_coh_old:.1f} cm")

