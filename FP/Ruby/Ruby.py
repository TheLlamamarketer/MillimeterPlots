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
Step_corrected = 0.59786924     # converts 0.6nm/1s steps to correct steps
Start_corrected = 346.671436
B_val = 765.0 # cm^-1  or 638.0

Hg_lines = np.array([365.0153, 404.6563, 435.8328, 253.6517*2, 546.0735, 576.9598, 579.0663, 296.7280*2, 312.5668*2, 313.16935000*2])


data: dict[str, np.ndarray] = {}


for file in sorted(source_dir.iterdir()):
    
    # Read first 3 lines separately
    with open(file, 'r') as f:
        start, step0, direction = [float(f.readline().strip()) for _ in range(3)]

    Ratio = Step_corrected / step0 if step0 == 0.6 else 10/6 * Step_corrected / step0 
    
    Offset = Start_corrected - Ratio * 350
    
    raw = pd.read_csv(
        file, header=None, decimal=".", sep=r"\s+", engine="python", skiprows=3).to_numpy(dtype=float)

    t = raw[:, 0]
    A = raw[:, 1]
    
    key = file.stem[2:] if not file.stem.startswith('Hg') else file.stem
    
    data[key] = {
        't': t, 
        'A': A, 
        'header': [start, step0, direction, Ratio],
        #'lambda': Offset + Ratio * (start + direction * step0 * t),
        'lambda': start + step0 * direction * t
    }

# --------------------------------------------------------------------------------
# --- Analyze Hg lines for calibration -------------------------------------------
# --------------------------------------------------------------------------------
data['Hg_lines']['A'] = -(data['Hg_lines']['A'] + np.min(-data['Hg_lines']['A']))          # invert and shift to zero baseline as we have emission lines


A_Hg = data['Hg_lines']['A']
A_Hg = A_Hg / np.max(A_Hg)  # normalize to max
data['Hg_lines']['A'] = A_Hg

lam_Hg = data['Hg_lines']['lambda']


def find_Hg_peaks(lam_Hg, A_Hg, Hg_lines):

    # find peaks
    peaks_hg, props_hg = find_peaks(A_Hg,  height=0.012)
    widths_hg = peak_widths(A_Hg, peaks_hg, rel_height=0.21)
    
    results_Hg_y = np.array([])
    line_peaks: list[tuple[float, float]] = [] 
    
    for i, peak_i in enumerate(peaks_hg):      
        half_win = 0.3748 * widths_hg[0][i]       

        fit_left  = max(0, int(np.floor(peak_i - half_win)))
        fit_right = min(len(lam_Hg) - 1, int(np.ceil(peak_i + half_win)))
        
        lam_per_peak = lam_Hg[fit_left:fit_right + 1]
        A_per_peak   = A_Hg[fit_left:fit_right + 1]
        
        
        delta_lambda = np.mean(np.diff(lam_per_peak)) 
        fwhm_lambda  = widths_hg[0][i] * delta_lambda
        sigma_init   = fwhm_lambda / 2.355
    
        result_Hg = lmfit(
            xdata= lam_per_peak,
            ydata= A_per_peak,
            model="gaussian",
            initial_params={'c': sigma_init, 'b': lam_Hg[peak_i]} # a * np.exp(-((x - b) / c)**2 / 2) + d
        )
        
        results_Hg_y = np.append(results_Hg_y, [result_Hg], axis=0)
        b_val = result_Hg.params['b'].value
        
        line_peaks.append((b_val, props_hg['peak_heights'][i]))

    lambda_peaks = np.array([peak[0] for peak in line_peaks])
    
    results_line_hg = lmfit(xdata=lambda_peaks, ydata=Hg_lines, model="linear")
    
    def get_cov(result, p1, p2):
        names = list(result.params.keys())
        i = names.index(p1)
        j = names.index(p2)
        return result.covar[i, j]
    
    cov_ab = get_cov(results_line_hg, 'a', 'b')


    s_cal = []
    for result_Hg, lambda_peak in zip(results_Hg_y, lambda_peaks):
        sa = results_line_hg.params['a'].stderr
        sb = results_line_hg.params['b'].stderr
        slam_nom = result_Hg.params['b'].stderr
        b = results_line_hg.params['b'].value

        var = (sa**2
            + (lambda_peak**2) * sb**2
            + (b**2) * slam_nom**2
            + 2 * lambda_peak * cov_ab)
        s_cal.append(np.sqrt(var))
        
    data_hg = {
        'linelist': Hg_lines,
        'fitted':  lambda_peaks,
        'dfitted': [result_Hg.params['b'].stderr for result_Hg in results_Hg_y],
        'calibrated': results_line_hg.eval(x=lambda_peaks),
        'diff': Hg_lines - results_line_hg.eval(x=lambda_peaks),
        'dcalibrated': s_cal,
    }

    headers_hg = {
        'linelist':     {'label': '{$\\lambda_{real}/nm$}', 'intermed': True },
        'fitted':       {'label': '{$\\lambda_{nom}/nm$}', 'intermed': False, 'err': data_hg['dfitted']},
        'calibrated':   {'label': '{$\\lambda_{fit}/nm$}', 'intermed': True, 'err': data_hg['dcalibrated']},
        'diff':         {'label': '{$\\lambda_{real} - \\lambda_{fit} /nm$}', 'intermed': True},
    }

    print_standard_table(
        data_hg,
        headers_hg,
        caption="Literaturwerte $\\lambda_{real}$ und gemessene Wellenlängen $\\lambda_{nom}$ der Quecksilberlinien sowie deren kalibrierte Werte $\\lambda_{fit}$ und Differenzen.",
        label="tab:ruby_hg_lines",
        show=True
    )
    
    print(f"RMS of differences: {np.sqrt(np.mean((results_line_hg.eval(x=lambda_peaks) - Hg_lines)**2)):.4g} nm")
    
    print(results_line_hg.fit_report())
    print(f"a = {print_round_val(results_line_hg.params['a'].value, results_line_hg.params['a'].stderr)} nm, b = {print_round_val(results_line_hg.params['b'].value, results_line_hg.params['b'].stderr)}")




    fighg, axhg = plot_data(
        [DatasetSpec(x=lambda_peaks, y=Hg_lines, label='Hg Lines from NIST', color='tab:blue', marker='o', fit_x=lambda_peaks, fit_y=results_line_hg.eval(x=lambda_peaks), fit_label='Linear Fit', fit_color='tab:cyan')],
        title=f'Hg Line Calibration',
        ylabel='$\\lambda_{real}$ (nm)',
        xlabel='$\\lambda_{nom}$ (nm)',
        legend_position=(0.02, 0.65),
        color_seed=89,
        plot='figure',
    )
    
    axhg.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = axhg.twinx()
    ax2.axhline(0, color='tab:orange', linestyle='--', label='linear fit')
    ax2.set_ylabel('Fit Residuals (nm)', color='tab:orange')
    ax2.plot(lambda_peaks, results_line_hg.eval(x=lambda_peaks) - Hg_lines, 'x', label='Fit Residuals', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.legend(loc='lower right')
    fighg.tight_layout()
    
    fighg.savefig(f'Plots/Hg_line_fit.pdf', bbox_inches="tight")
    plt.show()


    return peaks_hg, line_peaks, results_line_hg, lambda_peaks

peaks_hg, line_peaks, results_line_hg, lambda_peaks = find_Hg_peaks(lam_Hg, A_Hg, Hg_lines)

lambda_peaks_corr = results_line_hg.eval(x=lambda_peaks)

for key in data.keys():
    data[key]['lambda'] = results_line_hg.eval(x=data[key]['lambda'])

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
    title=f'Quecksilber Emissionslinien mit gefundenen Peaks',
    xlabel='$\\lambda_{nom}$ (nm)',
    ylabel='Intensität I (normalisiert)',
    filename=f'Plots/Hg_lines.pdf',
    height=15,
    color_seed=89,
    plot=False
)







# --------------------------------------------------------------------------------
# --- Plotting all other data ----------------------------------------------------
# --------------------------------------------------------------------------------




names = [ 'Xe', 'Ruby'] #'diff',

#data['Ruby_N']['A'] *= 1.02669405
data['diff_B2']['A'] *= 10.0


s_all = []
s_all_N = []
for key in sorted(data.keys()):
    if key == 'Hg_lines':
        continue
    prefix = key.split('_')[0]
    second = key.split('_')[1]
    data[key]['A'] = data[key]['A'] if prefix == 'Ruby' or second == 'Basis' else -data[key]['A']
    s = DatasetSpec(
        x=data[key]['lambda'],
        y=data[key]['A'],
        label=key,
        marker='None',
        line='-',
    )
    mask = data[key]['lambda'] >= 670.0
    s_N = DatasetSpec(
        x=data[key]['lambda'][mask],
        y=data[key]['A'][mask],
        label=key,
        marker='None',
        line='-',
    )
    s_all.append(s)
    s_all_N.append(s_N)


plot_data(
    s_all,
    title=f'All Spectra',
    xlabel='$\\lambda (nm)$',
    ylabel='Amplitude I',
    filename=f'Plots/All.pdf',
    color_seed=87,
    plot=False
)





plot_data(
    s_all_N,
    title=f'All Spectra (>= 670 nm)',
    xlabel='$\\lambda (nm)$',
    ylabel='Amplitude I',
    filename=f'Plots/All_N.pdf',
    color_seed=87,
    plot=False
)


def find_lambda_shift(lam_ref, I_ref, lam, I, lam_range=None, max_shift=1.0):
    lam_ref = np.asarray(lam_ref, float)
    I_ref   = np.asarray(I_ref,   float)
    lam     = np.asarray(lam,     float)
    I       = np.asarray(I,       float)

    s = np.argsort(lam_ref); lam_ref, I_ref = lam_ref[s], I_ref[s]
    s = np.argsort(lam);     lam,     I     = lam[s],     I[s]

    if lam_range is not None:
        lo, hi = lam_range
        mask_ref = (lam_ref >= lo) & (lam_ref <= hi)
        mask     = (lam     >= lo-max_shift) & (lam     <= hi+max_shift)
        lam_ref, I_ref = lam_ref[mask_ref], I_ref[mask_ref]
        lam,     I     = lam[mask],         I[mask]
    
    # normalize Data so that median is 0 and peak-to-peak amplitude is 1
    I_ref = (I_ref - np.median(I_ref)) / np.ptp(I_ref)
    I     = (I     - np.median(I))     / np.ptp(I)
    
    # interpolating functions for both so that we can evaluate at common points
    f_ref = interp1d(lam_ref, I_ref, kind='linear', bounds_error=False, fill_value=np.nan)
    f     = interp1d(lam,     I,     kind='linear', bounds_error=False, fill_value=np.nan)
    
    def mse(shift):
        lam_common = lam_ref
        y_ref = f_ref(lam_common)
        y     = f(lam_common - shift)
        mask  = ~np.isnan(y) & ~np.isnan(y_ref)
        if mask.sum() < 10:
            return 1e12
        r = y_ref[mask] - y[mask]
        
        return float(np.mean(r*r))
    
    res = minimize_scalar(mse, bounds=(-max_shift, max_shift), method='bounded')
    return float(res.x)









shifts = {key: 0.0 for key in data.keys()}  # cumulative shifts for each spectrum

# 1) Align Xe_B to Xe_N in 670–715 (Xe_N is the absolute reference)
shift_XeB = find_lambda_shift(data['Xe_N']['lambda'], data['Xe_N']['A'],
                              data['Xe_B']['lambda'], data['Xe_B']['A'],
                              lam_range=(670, 715),
                              max_shift=2.0)


data['Xe_B']['lambda'] += shift_XeB
shifts['Xe_B'] += shift_XeB

# 2) Align all BROAD spectra (*_B) to Xe_B in 450–500
for key in sorted(data.keys()):
    if not key.endswith('_B') and not key.endswith('_B2') and not key.endswith('_Basis'):
        continue
    if key in ['Xe_B', 'Hg_lines']:
        continue

    lam = data[key]['lambda']
    I   = data[key]['A']

    shift = find_lambda_shift(data['Xe_B']['lambda'], data['Xe_B']['A'],
                              lam, I,
                              lam_range=(450, 500),
                              max_shift=2.0)

    data[key]['lambda'] += shift
    shifts[key] += shift
    

# 3) Align all NARROW spectra (*_N) to their own *_B in 670–715
for key in sorted(data.keys()):
    if not key.endswith('_N') and not key.endswith('_N2'):
        continue
    if key == 'Xe_N':   # Xe_N is the reference, do NOT shift
        continue

    broad_key = key.split('_')[0] + '_B'

    lam = data[key]['lambda']
    I   = data[key]['A']

    shift = find_lambda_shift(data[broad_key]['lambda'], data[broad_key]['A'],
                              lam, I,
                              lam_range=(670, 715),
                              max_shift=1.0)

    data[key]['lambda'] += shift
    shifts[key] += shift

# Print shifts as a table
data_shifts = {key: [shift] for key, shift in shifts.items() if key != 'Hg_lines'}
data_shifts = dict(sorted(data_shifts.items()))

headers_shifts = {key: {'label': f'{{ $ {key.split('_')[0] + "_{" + key.split('_')[1] + "}" } $}}', 'intermed': True} for key in data_shifts.keys()}

print_standard_table(
    data_shifts,
    headers_shifts,
    caption="Wellenlängenverschiebungen der Spektren nach der Ausrichtung, basierend auf der Minimierung der mittleren quadratischen Abweichung. Vergleichsspektrum ist das $Xe_N$ Spektrum.",
    label="tab:ruby_wavelength_shifts",
    show=True
)

s_all = []
for key in sorted(data.keys()):
    if key == 'Hg_lines':
        continue
    s = DatasetSpec(
        x=data[key]['lambda'],
        y=data[key]['A'],
        label=key,
        marker='None',
        line='-',
    )
    s_all.append(s)

plot_data(
    s_all,
    title=f'All Spectra',
    xlabel='$\\lambda (nm)$',
    ylabel='Amplitude I',
    filename=f'Plots/All.pdf',
    color_seed=87,
    plot=False
)

print('-'*100)


specs_by_key: dict[str, DatasetSpec] = {}
for key in sorted(data.keys()):
    if key == 'Hg_lines' or key == 'Xe_Basis':
        continue
    prefix = key.split('_')[0]
    bkey = f"{prefix}_B"
    is_ruby = prefix == 'Ruby'

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



def resample_to_grid(lam, I, grid):
    lam = np.asarray(lam, float)
    I   = np.asarray(I,   float)
    s = np.argsort(lam)
    lam, I = lam[s], I[s]
    f = interp1d(lam, I, kind='linear', bounds_error=False, fill_value=np.nan)
    return f(grid)

def compute_ratio(lam_num, I, lam_den, I0, eps=1e-12, ngrid=None):
    lam_num = np.asarray(lam_num, float)
    lam_den = np.asarray(lam_den, float)
    I       = np.asarray(I,       float)
    I0      = np.asarray(I0,      float)

    lo = max(lam_num.min(), lam_den.min())
    hi = min(lam_num.max(), lam_den.max())
    if hi <= lo:
        return np.array([]), np.array([])

    n = int(ngrid or max(len(lam_num), len(lam_den)))
    grid = np.linspace(lo, hi, n)

    num_i = resample_to_grid(lam_num, I,  grid)
    den_i = resample_to_grid(lam_den, I0, grid)

    mask  = (np.abs(den_i) > eps) & ~np.isnan(num_i) & ~np.isnan(den_i)
    if mask.sum() == 0:
        return np.array([]), np.array([])

    T = num_i[mask] / den_i[mask]
    T = np.clip(T, eps, None) 
    return grid[mask], T, num_i[mask], den_i[mask]



lam_diff_B, T_diff_B, I_diff_B, I_Xe_B = compute_ratio(data['diff_B']['lambda'], data['diff_B']['A'], data['Xe_B']['lambda'], data['Xe_B']['A'])
lam_B, T_B, I_Ruby_B, I_Xe_B2 = compute_ratio(data['Ruby_B']['lambda'], data['Ruby_B']['A'], data['Xe_B']['lambda'], data['Xe_B']['A'], eps=1e-4)



plot_data(
    [DatasetSpec(x=data['diff_B']['lambda'], y=data['diff_B']['A'], label='diff_B', marker='None', line='-'),
     DatasetSpec(x=data['Xe_B']['lambda'], y=data['Xe_B']['A'], label='Xe_B', marker='None', line='-'),
     DatasetSpec(x=data['Ruby_B']['lambda'], y=data['Ruby_B']['A'], label='Ruby_B', marker='None', line='-')],
    title=f'Vergleich der Spektren diff_B, Xe_B und Ruby_B',
    xlabel='$\\lambda$ (nm)',
    ylabel='Intensität I',
    filename=f'Plots/Raw_Ruby_Diff.pdf',
    height=15,
    plot=False
)

lam = np.linspace(350, 715, 10000)
plot_data(
    [DatasetSpec(x=lam_diff_B, y=1 - I_diff_B/I_Xe_B, label='diff_B / Xe_B', marker='None', line='-', fit_x=lam, fit_y=make_splrep(lam_diff_B, 1 - I_diff_B/I_Xe_B, s=0.5)(lam), fit_color='orange'),
     DatasetSpec(x=lam_B, y=T_B, label='Ruby_B / Xe_B', marker='None', line='-')],
    title=f'Transmittanzspektren von Rubin und Differenzspektrum',
    xlabel='$\\lambda$ (nm)',
    ylabel='Transmittanz T',
    filename=f'Plots/Transmittance_Ruby_Diff.pdf',
    height=15,
    plot=False
)

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

print(f"Resulting centers: $\\lambda_{{^4 T_1}} = ({results_diff_T_1.params['b'].value:.2f} \\pm {results_diff_T_1.params['b'].stderr:.2f})$ nm, $\\lambda_{{^4 T_2}} = ({results_diff_T_2.params['b'].value:.2f} \\pm {results_diff_T_2.params['b'].stderr:.2f})$ nm")
print(f"Resulting normed Emergies by B: $E_{{^4 T_1}} = ({const/ results_diff_T_1.params['b'].value:.3f} \\pm {const * results_diff_T_1.params['b'].stderr / (results_diff_T_1.params['b'].value**2):.3f})$, $E_{{^4 T_2}} = ({const/ results_diff_T_2.params['b'].value:.3f}  \\pm {const * results_diff_T_2.params['b'].stderr / (results_diff_T_2.params['b'].value**2):.3f})$ with ratio {results_diff_T_2.params['b'].value / results_diff_T_1.params['b'].value:.4f}")


plot_data(
    [s_B],
    title=f'Absorptionsspektrum von Rubin',
    xlabel='$\\lambda$ (nm)',
    ylabel='Absorbanz (A)',
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

# Determine asymmetric Dq/B errors by inverting at E ± dE
Dq_T1_plus  = fsolve(lambda Dq: spline_T1(Dq) - (E_T1 + dE_T1), x0=Dq_T1)[0]
Dq_T1_minus = fsolve(lambda Dq: spline_T1(Dq) - (E_T1 - dE_T1), x0=Dq_T1)[0]
dDq_T1_up   = max(0.0, Dq_T1_plus  - Dq_T1)
dDq_T1_dn   = max(0.0, Dq_T1       - Dq_T1_minus)

Dq_T2_plus  = fsolve(lambda Dq: spline_T2(Dq) - (E_T2 + dE_T2), x0=Dq_T2)[0]
Dq_T2_minus = fsolve(lambda Dq: spline_T2(Dq) - (E_T2 - dE_T2), x0=Dq_T2)[0]
dDq_T2_up   = max(0.0, Dq_T2_plus  - Dq_T2)
dDq_T2_dn   = max(0.0, Dq_T2       - Dq_T2_minus)

print(f"Fitted Energy Levels by B:\n"
    f"E_T1 = {E_T1:.3f} ± {dE_T1:.3f} at Dq/B = {Dq_T1:.5f} (+{dDq_T1_up:.5f}/-{dDq_T1_dn:.5f}),\n "
    f"E_T2 = {E_T2:.3f} ± {dE_T2:.3f} at Dq/B = {Dq_T2:.5f} (+{dDq_T2_up:.5f}/-{dDq_T2_dn:.5f}),\n"
    f"with ratio E_T2 / E_T1 = {E_T2 / E_T1:.4f}")
print(f"Resulting 10 Dq from fits: 10 Dq = {10* Dq_T1 * B_val:.2f} cm^-1 or {10 * Dq_T1 * B_val * 1.23981e-4:.4f} eV from T1, 10 Dq = {10 * Dq_T2 * B_val:.2f} cm^-1 or {10 * Dq_T2 * B_val * 1.23981e-4:.4f} eV from T2")

print(f"Resultierendes 10 Dq mit Fehlern aus dem Intervall mit durchschnittlichen Abweichungen:\n"
    f"10 Dq = {10* np.mean([Dq_T1, Dq_T2]) * B_val:.2f} cm^-1 \\pm {np.abs(10 * (Dq_T1 - Dq_T2) * B_val / 2):.2f} cm^{{-1}} \n"
    f"oder {10 * np.mean([Dq_T1, Dq_T2]) * B_val * 1.23981e-4:.4f} eV \\pm {np.abs(10 * (Dq_T1 - Dq_T2) * B_val * 1.23981e-4 / 2):.4f} eV")

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
    plot=False
)



