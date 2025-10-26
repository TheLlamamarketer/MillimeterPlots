import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from decimal import Decimal

from Functions.plotting import plot_data, DatasetSpec  
from Functions.tables import print_standard_table, print_complex_table, datasets_to_table_blocks
from Functions.help import * 
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


data = {
    'dI_S': [0.0005], 'dT': [0.5], 'd': 1e-3,
    '1': {
        'a': {
            'U_H':  ['7.670','8.929','10.083','11.199','12.427','13.674','15.059','16.274','17.613','18.988','20.167','21.450','22.805','24.020','25.270'],
            'U_H_0':['-1.280'],
            'I_B':  ['0.498','0.584','0.657','0.725','0.800','0.874','0.954','1.026','1.103','1.183','1.251','1.326','1.405','1.476','1.550'],
            'dU_H': [], 'dI_B': [],
            'I_S':  ['20'], 'B': [], 'dB': [],
        },
        'b': {
            'U_H_0': ['-1.268','-1.335','-1.366','-1.410','-1.466','-1.555','-1.595','-1.662','-1.711','-1.744','-1.791','-1.831','-1.877','-1.936','-2.013','-2.089'],
            'I_S':   ['20','19','18','17','16','14','13','11','10','9','8','7','6','5','3','2'],
            'U_H':   ['25.234','24.320','22.484','21.360','19.652','16.840','15.388','12.415','10.714','9.164','7.232','6.930','5.066','3.737','0.750','-1.105'],
            'I_B':   ['1.529'],
            'dU_H': [], 'dI_B': [], 'dU_H_0': [], 'B': [], 'dB': [],
        },
    },
    '2': {
        'T':  [125,124,123,122,121,120,119,118,117,116,115,114,113,112,111,110,109,108,107,106,
               105,104,103,102,101,100,99,98,97,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,
               81,80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,65,64,63,62,61,60,59,58,57,
               56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,
               31,30,29,28,27],
        'U_H': ['5.090','5.243','5.424','5.621','5.826','6.031','6.213','6.496','6.774','6.954','7.180',
                '7.455','7.667','7.885','8.226','8.570','8.837','9.096','9.455','9.736','10.022','10.432',
                '10.75','11.112','11.439','11.772','12.117','12.509','12.936','13.287','13.684','14.093',
                '14.496','14.869','15.281','15.696','16.055','16.420','16.791','17.217','17.624','18.010',
                '18.333','18.765','19.089','19.458','19.762','20.126','20.467','20.776','21.040','21.335',
                '21.623','21.865','22.115','22.344','22.543','22.724','22.938','23.136','23.312','23.488',
                '23.642','23.782','23.898','24.032','24.154','24.246','24.365','24.448','24.532','24.605',
                '24.674','24.735','24.798','24.845','24.893','24.935','24.971','24.999','25.027','25.050',
                '25.070','25.090','25.107','25.118','25.127','25.133','25.135','25.138','25.135','25.133',
                '25.133','25.130','25.128','25.129','25.131','25.142','25.185'],
        'dU_H': ["0.107","0.139","0.183","0.17","0.2","0.15","0.161","0.32","0.116","0.207","0.233","0.166",
                 "0.201","0.22","0.369","0.228","0.186","0.273","0.243","0.203","0.286","0.292","0.24","0.358",
                 "0.251","0.328","0.297","0.274","0.4","0.298","0.334","0.351","0.328","0.194","0.375","0.336",
                 "0.385","0.313","0.333","0.36","0.384","0.285","0.259","0.255","0.292","0.27","0.278","0.313",
                 "0.302","0.199","0.247","0.257","0.2","0.215","0.197","0.19","0.178","0.178","0.17","0.161",
                 "0.178","0.166","0.138","0.14","0.068","0.089","0.062","0.099","0.083","0.07","0.071","0.063",
                 "0.039","0.057","0.051","0.04","0.038","0.041","0.025","0.026","0.021","0.018","0.019","0.017",
                 "0.013","0.008","0.006","0.004","0.002","0.002","0.005","0.001","0.001","0.006","0.001","0",
                 "0.005","0.012","0.026"]
    }
}


def calc_dU(U):
    return abs(np.array(U, dtype=float) * 0.004) + 5 * last_digit(U)

def calc_dI_B(I_B):
    return abs(np.array(I_B, dtype=float) * 0.008) + 5 * last_digit(I_B)

def calc_B(I):
    return I*0.13636 + 0.004546

def _at_safe(arr, i, default):
    try:
        return arr[i]
    except Exception:
        return default

# -------------------------------------------------------------------------------
# --- Data Preparation ----------------------------------------------------------
# -------------------------------------------------------------------------------


for key in data['1']:
    U_0 = np.array([Decimal(U) for U in data['1'][key]['U_H_0']], dtype=object) if 'U_H_0' in data['1'][key] else np.array([Decimal('0')], dtype=object)
    U_H = np.array([Decimal(U) for U in data['1'][key]['U_H']], dtype=object)
    I_B = np.array([Decimal(I) for I in data['1'][key].get('I_B', ['0'])], dtype=object)
    I_S = np.array([Decimal(I) for I in data['1'][key]['I_S']], dtype=object)

    # uncertainties
    dU_H_corr = np.sqrt(calc_dU(U_H.astype(float))**2 + calc_dU(U_0.astype(float))**2) / 1000.0
    dI_B_val  = calc_dI_B(I_B.astype(float))

    # apply zero correction and convert units
    U_H_corr = (U_H - _at_safe(U_0, 0, Decimal('0')))  # subtract first U0 for set (ok for a), for (b) U0 array is same length
    if len(U_0) == len(U_H):  # (b) has a vector U_H_0: subtract element-wise
        U_H_corr = U_H - U_0

    data['1'][key]['dU_H'] = np.array(dU_H_corr, dtype=float)
    data['1'][key]['dI_B'] = np.array(dI_B_val, dtype=float)

    data['1'][key]['U_H']  = np.array(U_H_corr, dtype=float) / 1000.0   # V
    data['1'][key]['I_B']  = np.array(I_B, dtype=float)                 # A
    data['1'][key]['I_S']  = np.array(I_S, dtype=float) / 1000.0        # A

    data['1'][key]['B']   = calc_B(data['1'][key]['I_B'])               # T
    data['1'][key]['dB']  = np.sqrt((0.13636 * data['1'][key]['dI_B'])**2 + 0.005**2)  # T


headers_a = {
    'U_H': {'label': '{$U_H/mV$}', 'err': data['1']['a']['dU_H'] * 1000, 'data': data['1']['a']['U_H'] * 1000},
    'I_B': {'label': '{$I_B/A$}',  'err': data['1']['a']['dI_B'],        'data': data['1']['a']['I_B']},
    'B':   {'label': '{$B/mT$}',   'err': data['1']['a']['dB'] * 1000,   'data': data['1']['a']['B'] * 1000},
}

print_standard_table(
    data=data['1']['a'],
    headers=headers_a,
    column_formats=["2.1","1.3","2.2"],
    caption="A1a: corrected Hall voltage, magnet current, and field",
    label="tab:A1a",
    show=False
)
headers_b = {
    'U_H': {'label': '{$U_H/mV$}', 'err': data['1']['b']['dU_H'] * 1000, 'data': data['1']['b']['U_H'] * 1000},
    'I_S': {'label': '{$I_S/mA$}', 'err': data['dI_S'][0] * 1000,        'data': data['1']['b']['I_S'] * 1000},
}


print_standard_table(
    data=data['1']['b'],
    headers=headers_b,
    column_formats= ["2.1"] * len(headers_b),
    caption="A1b: source current vs corrected Hall voltage",
    label="tab:A1b",
    show=False
)

# -------------------------------------------------------------------------------
# --- Analysis ------------------------------------------------------------------
# -------------------------------------------------------------------------------

result_1_I = lmfit(data['1']['a']['U_H'], data['1']['a']['B'], data['1']['a']['dB'])
result_2_I = lmfit(data['1']['b']['U_H'], data['1']['b']['I_S'], data['dI_S']*np.ones(len(data['1']['b']['I_S'])))
result_1_U = lmfit(data['1']['a']['B'], data['1']['a']['U_H'], data['1']['a']['dU_H'])
result_2_U = lmfit(data['1']['b']['I_S'], data['1']['b']['U_H'], data['1']['b']['dU_H'])


a1, da1, b1, db1, _, _ = slope(data['1']['a']['U_H'], data['1']['a']['B'], data['1']['a']['dB'])
a2, da2, b2, db2, _, _ = slope(data['1']['b']['U_H'], data['1']['b']['I_S'], data['dI_S']*np.ones(len(data['1']['b']['I_S'])))
print(f'b = {print_round_val(result_1_I.params['b'].value, result_1_I.params['b'].stderr)}T/V')
print(f'b = {print_round_val(result_2_I.params['b'].value, result_2_I.params['b'].stderr)}A/V')


B0 = data['1']['b']['B']
R_H_a = data['d']/ result_1_I.params['b'].value/ data['1']['a']['I_S'][0]*1e3
R_H_b = data['d']/ result_2_I.params['b'].value/ B0[0]*1e3

dR_H_a = R_H_a * np.sqrt( (result_1_I.params['b'].stderr/result_1_I.params['b'].value)**2 + (data['dI_S'][0]/data['1']['a']['I_S'][0])**2 )
dR_H_b = R_H_b * np.sqrt( (result_2_I.params['b'].stderr/result_2_I.params['b'].value)**2 + (data['1']['b']['dB'][0]/B0[0])**2 )

print(f'R_H_a = ({round_val(R_H_a, dR_H_a)[0]} \\pm {round_val(R_H_a, dR_H_a)[1]}) 10^{{-3}} \\ m^3/C')
print(f'R_H_b = ({round_val(R_H_b, dR_H_b)[0]} \\pm {round_val(R_H_b, dR_H_b)[1]}) 10^{{-3}} \\ m^3/C')

R_H_a_slope = data['d']/ b1/ data['1']['a']['I_S'][0]*1e3
R_H_b_slope = data['d']/ b2/ B0[0]*1e3
dR_H_a_slope = R_H_a_slope * np.sqrt( (db1/b1)**2 + (data['dI_S'][0]/data['1']['a']['I_S'][0])**2 )
dR_H_b_slope = R_H_b_slope * np.sqrt( (db2/b2)**2 + (data['1']['b']['dB'][0]/B0[0])**2 )

print(f'R_H_a_slope = ({round_val(R_H_a_slope, dR_H_a_slope)[0]} \\pm {round_val(R_H_a_slope, dR_H_a_slope)[1]})10^{{-3}}m^3/C')
print(f'R_H_b_slope = ({round_val(R_H_b_slope, dR_H_b_slope)[0]} \\pm {round_val(R_H_b_slope, dR_H_b_slope)[1]})10^{{-3}}m^3/C')


# -------------------------------------------------------------------------------
# --- Plots 1 -------------------------------------------------------------------
# -------------------------------------------------------------------------------


x_line = np.linspace(data['1']['a']['U_H'][0], data['1']['a']['U_H'][-1], 100)


s1 = DatasetSpec(x=data['1']['a']['U_H'], y=data['1']['a']['B'], xerr=data['1']['a']['dU_H'], yerr=data['1']['a']['dB'],
                 fit_x=x_line, fit_y=result_1_I.eval(x=x_line), confidence=calc_CI(result_1_I, x_line, 3),
                 label='Messdaten', color_group='1')

plot_data(
    filename='Plots/HAL_I1.pdf',  # CHANGED: filename first
    datasets=s1,
    ylabel=r'$B \ \mathrm{[T]}$',
    xlabel=r'$U_H \ \mathrm{[V]}$',
    title='Hallspannung im Magnetfeld',
    plot=False
)


x_line = np.linspace(data['1']['b']['U_H'][0], data['1']['b']['U_H'][-1], 100)

plot_data(
    datasets=DatasetSpec(x=data['1']['b']['U_H'], y=data['1']['b']['I_S'], xerr=data['1']['b']['dU_H'], yerr=data['dI_S']*np.ones(len(data['1']['b']['I_S'])),
                 fit_x=x_line, fit_y=result_2_I.eval(x=x_line), confidence=calc_CI(result_2_I, x_line, 3), label='Messdaten', color_group='1'),
    ylabel=r'$I_S \ A$',
    xlabel=r'$U_H \ V$',
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_I2.pdf',
    plot=False
)


# -------------------------------------------------------------------------------
# --- Bandgap Energy -----------------------------------------------------------
# -------------------------------------------------------------------------------


U_H2 = np.array([Decimal(U) for U in data['2']['U_H']], dtype=object)
T2   = np.array([Decimal(T) for T in data['2']['T']], dtype=object)
dU_H2= np.array([Decimal(U) for U in data['2']['dU_H']], dtype=object)


data['2']['U_H'] = np.array(U_H2, dtype=float) / 1000.0   
data['2']['T']   = np.array(T2,   dtype=float) + 273.15    
data['2']['dU_H']= np.maximum(calc_dU(U_H2.astype(float))/1000.0, dU_H2.astype(float)/1000.0/np.sqrt(2))


y = np.log(data['2']['U_H']**2 * data['2']['T']**3)
x = 1.0 / data['2']['T']
dx = data['dT'] * np.ones(len(data['2']['T']))/data['2']['T']**2
dy = np.sqrt((2*data['2']['dU_H']/data['2']['U_H'])**2 + (3 * data['dT'][0] * np.ones(len(data['2']['T']))/data['2']['T'])**2)


spline = UnivariateSpline(x, y, s=0.0025, k=4)
spline_der = spline.derivative()
spline_der2 = spline.derivative(n=2)

x_smooth = np.linspace(np.min(x), np.max(x), 300)
x_smooth_reduced = np.linspace(np.min(x[:-55]), np.max(x[:-55]), 300)

m_spline = spline_der(x[np.argmin(spline_der2(x))])

# index size between each point 0.0000085 1/K 


idx1 = np.where((x >= 0.002651) & (x <= 0.002744))
idx2 = np.where((x >= 0.002556) & (x <= 0.002701))

result  = lmfit(x[idx1], y[idx1], dy[idx1])
result2 = lmfit(x[idx2], y[idx2], dy[idx2])

k_b = 1.380649e-23
e = 1.602176634e-19

E_1 = round_val(result.params['b'].value*k_b/e, result.params['b'].stderr*k_b/e)
print(f"$E_1 = {E_1[0]} \\pm {E_1[1]}$")
E_2 = round_val(result2.params['b'].value*k_b/e, result2.params['b'].stderr*k_b/e)
print(f"$E_2 = {E_2[0]} \\pm {E_2[1]}$")
print(f"m_spline = {m_spline*k_b/e}")


E = (E_1[0]/E_1[1]**2 + E_2[0]/E_2[1]**2)/(1/E_1[1]**2 + 1/E_2[1]**2)
dE = 1/np.sqrt(1/E_1[1]**2 + 1/E_2[1]**2)
print(f'$E_g = {print_round_val(E, dE, False)} eV$')


# -------------------------------------------------------------------------------
# --- Plotting 2 ----------------------------------------------------------------
# -------------------------------------------------------------------------------


def plot_intervals_r2(x, y, dy, min_window_size=5, max_window_size=20):
    colors = plt.cm.viridis(np.linspace(0, 1, max_window_size - min_window_size + 1))
    datasets = []
    best_intervals = []

    for window_size, color in zip(range(min_window_size, max_window_size + 1), colors):
        intervals = []
        r2_values = []
        b_values = []
        n_points = len(x)
        for start in range(n_points - window_size):
            end = start + window_size
            result = lmfit(x[start:end], y[start:end], dy[start:end])
            r2 = result.rsquared
            b = result.params['b'].value
            intervals.append(x[start])
            r2_values.append(r2)
            b_values.append(b)
        
        best_index = np.argmax(r2_values)
        best_intervals.append((window_size, intervals[best_index], r2_values[best_index], b_values[best_index]))

        datasets.append({
            'ydata': r2_values,
            'xdata': intervals,
            'label': f'Window size {window_size}',
            'color': color
        })
    
    plot_data(
        datasets=datasets,
        ylabel=r'$R^2$',
        xlabel=r'Starting value of interval',
        ymin=0.99,
        ymax=1,
        title='R^2 values for different intervals',
        filename=f'Plots/R2_intervals.pdf',
        plot=False
    )

    # Evaluate the best intervals based on a score
    maxr2 = max([r2 for window_size, start, r2, b in best_intervals])
    minr2 = min([r2 for window_size, start, r2, b in best_intervals])
    maxwindow = max([window_size for window_size, start, r2, b in best_intervals])
    scores = [(window_size, start, r2, (r2-minr2)/(maxr2-minr2) + window_size/maxwindow, b) for window_size, start, r2, b in best_intervals]
    scores.sort(key=lambda x: x[3], reverse=True)

    for window_size, start, r2, score, b in scores:
        interval = window_size/((125+273.15)*(24+273.15))
        print(f'Window size: {window_size}, Start: {start:.6f}, End: {(start + interval):6f}, Interval: {interval:6f}, R^2: {r2:.4f}, Score: {score:.4f}, slope: {b:.4f}')

#plot_intervals_r2(x[:-55], y[:-55], dy[:-55], min_window_size=5, max_window_size=28)



plot_data(
    datasets= DatasetSpec(x=data['2']['T'], y=data['2']['U_H'], xerr=data['dT']*np.ones(len(data['2']['T'])), yerr=data['2']['dU_H']),
    ylabel=r'$U_H \ V$',
    xlabel=r'$T \ K$',
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_2pre.pdf',
    plot=False
)

s1 = DatasetSpec(x=x, y=y, xerr=dx, yerr=dy,fit_x=x_smooth, fit_y=spline(x_smooth),label='Messdaten', color_group='1')
s2 = DatasetSpec(x=x, y=result.eval(x=x), marker="None", line="-", label="Linearer Fit 1", color_group='2',
                 confidence=calc_CI(result, x, 3), fit_label=False)
s3 = DatasetSpec(x=x, y=result2.eval(x=x), marker="None", line="-", label="Linearer Fit 2", color_group='3',
                 confidence=calc_CI(result2, x, 3), fit_label=False)
s4 = DatasetSpec(x=x_smooth, y=spline_der(x_smooth[np.argmin(spline_der2(x_smooth))])*(x_smooth-x_smooth[np.argmin(spline_der2(x_smooth))]) + spline(x_smooth[np.argmin(spline_der2(x_smooth))]),
                 marker="None", line="--", label="Tangente an Wendepunkt", color_group='2', fit_label=False)

plot_data(
    datasets= [s1, s2, s3],
    ylabel=r'$ln(U_H^2 \cdot T^3)$',
    xlabel=r'$1/T \ K^{-1}$',
    ylim=(None, 10),
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_2.pdf',
    plot=False,
)

d1 = DatasetSpec(x=x[:-55], y=y[:-55], xerr=dx[:-55], yerr=dy[:-55],fit_x=x_smooth_reduced, fit_y=spline(x_smooth_reduced),label='Messdaten', color_group='1')
d2 = DatasetSpec(x=x[:-55], y=result.eval(x=x[:-55]), marker="None", line="-", label="Linearer Fit 1", color_group='2', confidence=calc_CI(result, x[:-55], 3), fit_label=False)
d3 = DatasetSpec(x=x[:-55], y=result2.eval(x=x[:-55]), marker="None", line="-", label="Linearer Fit 2", color_group='3', confidence=calc_CI(result2, x[:-55], 3), fit_label=False)
d4 = DatasetSpec(x=x_smooth_reduced, y=spline_der(x_smooth_reduced[np.argmin(spline_der2(x_smooth_reduced))])*(x_smooth_reduced-x_smooth_reduced[np.argmin(spline_der2(x_smooth_reduced))]) + spline(x_smooth_reduced[np.argmin(spline_der2(x_smooth_reduced))]),
                 marker="None", line="--", label="Tangente an Wendepunkt", color_group='2', fit_label=False)

plot_data(
    datasets= [d1, d2, d3],
    ylabel=r'$ln(U_H^2 \cdot T^3)$',
    xlabel=r'$1/T \ [1/K]$',
    ylim=(None, 10),
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_2close.pdf',
    plot=False,
)

plot_data(
    datasets= DatasetSpec(x=x, y=y - spline(x), xerr=dx, yerr=dy),
    ylabel=r'$3 \ ln(U_H \cdot T) - y$',
    xlabel=r'$1/T \ [1/K]$',
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_2_res.pdf',
    plot=False
)

plot_data(
    datasets= DatasetSpec(x=x_smooth, y=spline_der(x_smooth)*k_b/e, line='-', marker="None"),
    ylabel=r'$\frac{d}{dT} 3 \ ln(U_H \cdot T) $',
    xlabel=r'$1/T \ [1/K]$',
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_2_der.pdf',
    plot=False
)

plot_data(
    datasets= DatasetSpec(x=x_smooth, y=spline_der2(x_smooth), line='-', marker="None"),
    ylabel=r'$\frac{d^2}{dT^2} 3 \ ln(U_H \cdot T) $',
    xlabel=r'$1/T \ [1/K]$',
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_2_der2.pdf',
    plot=False
)
