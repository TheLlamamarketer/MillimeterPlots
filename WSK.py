from matplotlib.pylab import f
import numpy as np
from plotting import plot_data
from tables import *
from help import *
from decimal import Decimal

data = {
    '1': {
        't': np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19.6, 11.2, 0.7, 1.6, 9.4, 3.6, 8.9, 8.8]),
        'U': np.array([0.5, 1.7, 2.05, 2.15, 2.18, 1.05, 0.30, 0.10, 0.05, 0, 0, 0.5, 1.05, 1.5, 1.5, 2, 2, 2.18]),
        'R': 17.81e3,
        'C': 0.1057e-6,
    },
    '2': {
        'R': 8.5,
        'HP': {
            'f': np.array([10.04, 25.84, 40.10, 50.11, 100.15, 250.0, 500, 1.0015e3, 1.508e3, 2.0077e3, 3.0001e3, 3.5015e3, 5.0503e3, 7.5020e3, 10.093e3, 15.083e3, 25.05e3]),
            'U_0': np.array([0.79e3, 0.804e3, 0.805e3, 0.804e3, 0.799e3, 0.768e3, 0.681e3, 499.3, 376.7, 299.8, 213, 186, 134.4, 94.1, 71.3, 47.6, 25.2]),
            'U': np.array([1.5, 3.8, 5.8, 7.3, 14.4, 34.2, 60.4, 87.4, 97.1, 100, 99, 96.9, 88, 73.3, 60.4, 43, 23.5]),
            'df': np.array([0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.05, 0.001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.1e3]),
            'dU': np.array([0.2023, 0.2057, 0.2087, 0.2109, 0.2216, 0.2513, 0.2906, 0.3311, 0.3457, 0.3500, 0.3485, 0.3454, 0.3320, 0.3100, 0.2906, 0.2645, 0.2353]),
            'dU_0': np.array([10, 3.2060, 3.2075, 3.2060, 3.1985, 3.1520, 3.0215, 0.9489, 0.7651, 0.6497, 2.3195, 2.2790, 0.4016, 0.3412, 0.3070, 0.2714, 0.2378]),
            'C': 3.354e-6,
        },
        'TP': {
            'f': np.array([10.01, 25.07, 40.19, 50.01, 100.15, 250.05, 405.55, 505.6, 1007.4, 1.504e3, 2.009e3, 3.002e3, 3.503e3, 5.007e3, 7.507e3, 10.073e3, 15.038e3, 24.985e3, 50.02e3]),
            'U_0': np.array([155, 158.9, 159.6, 160, 163.2, 183.1, 215.8, 240.1, 366.2, 466.7, 536.2, 602.1, 611.1, 593, 514.4, 434.3, 320.6, 199.8, 89.6]),
            'U': np.array([103, 105.7, 105.8, 106, 105.8, 105.2, 104.2, 103.1, 95.1, 84.9, 74.4, 56.8, 49.7, 34.2, 20.1, 12.4, 4.4, 1.3, 0.2]),
            'dU': np.array([1, 0.3586, 0.3587, 0.3587, 0.3587, 0.3578, 0.3563, 0.3547, 0.3427, 0.3274, 0.3116, 0.2852, 0.2746, 0.2513, 0.2302, 0.2186, 0.2066, 0.2020, 0.2003]),
            'dU_0': np.array([2, 0.4384, 0.4394, 0.4400, 0.4448, 0.4747, 0.5237, 0.5602, 0.7493, 0.9001, 1.0043, 1.1032, 1.1167, 2.8895, 0.9716, 0.8515, 0.6809, 0.4997, 0.3344]),
            'L': 4.720e-3, 'dL': 0.05e-3, 'R_L': 4
        },
        'BP': {
            'f': np.array([10.035, 25, 40.075, 50.015, 100.05, 249.9, 500, 1.005e3, 1.507e3, 2.004e3, 3.002e3, 3.506e3, 5.019e3, 7.498e3, 10.076e3, 15.027e3, 25.08e3, 749.3, 1.2508e3]),
            'U_0': np.array([0.775e3, 0.72e3, 0.642e3, 0.591e3, 403, 213.5, 147.9, 126.4, 127.1, 133, 149.6, 158.1, 180.4, 201.5, 207.5, 196.4, 153.8, 131.5, 125.7]),
            'U': np.array([20.6, 46.2, 64.4, 73.1, 95.4, 107.8, 110.3, 110.1, 108.5, 106.3, 100.6, 97.2, 86.0, 68.6, 53.9, 34.4, 13.6, 110.3, 109.3]),
            'dU':np.array([0.2309, 0.2693, 0.2966, 0.3096, 0.3431, 0.3617, 0.3654, 0.3651, 0.3628, 0.3595, 0.3509, 0.3458, 0.3290, 0.3029, 0.2809, 0.2516, 0.2204, 0.3654, 0.3639]),
            'dU_0': np.array([5, 1.0800, 2.9630, 2.8865, 2.6045, 0.3202, 0.2219, 0.1896, 0.1906, 2.1995, 0.2244, 0.2371, 0.2706, 0.3023, 0.3113, 0.2946, 0.2307, 0.1973, 0.1885]),
            'C': 42.49e-6, 'L': 4.839e-6, 'R_L': 0.65, 'dR_L': 0.05,
        }
    }
}

def last_digit(num, n=1):
    if isinstance(num, np.ndarray):
        return np.vectorize(last_digit)(num, n)
    elif isinstance(num, (int, float, Decimal)):
        if isinstance(num, (float, Decimal)):
            return  round(n * 10**-LastSignificant(num), LastSignificant(num))
        else: 
            return n
    else:
        raise TypeError("Input must be an int, float, Decimal or np.ndarray")

def calc_dU(U):
    return np.array(U, dtype=float) * 0.0015 + last_digit(U, 2)

#U = ["1.5", "3.8", "5.8", "7.3", "14.4", "34.2", "60.4", "87.4", "97.1", "100.0", "99.0", "96.9", "88.0", "73.3", "60.4", "43.0", "23.5"]
#U = np.array([Decimal(u) for u in U])

#print(", ".join(map(lambda x: f"{x:.4f}", calc_dU(U))))

sorted_indices = np.argsort(data['1']['t'])
data['1']['t'] = data['1']['t'][sorted_indices]
data['1']['U'] = data['1']['U'][sorted_indices]


char_mask = data['1']['t'] <= 8.8
dischar_mask = data['1']['t'] >= 8.8

result_1 = linear_fit(data['1']['t'][char_mask], data['1']['U'][char_mask], model='exponential_decay', constraints={"a": {"min": 0}, "b": {"min": 0}})

print(extract_params(result_1)['a'][0], extract_params(result_1)['b'][0])

result_2 = linear_fit(data['1']['t'][dischar_mask]-8.8, data['1']['U'][dischar_mask], model='exponential', constraints={"a": {"min": 0}, "b": {"max": -0.58}, "c":0})

print(extract_params(result_2)['a'][0], extract_params(result_2)['b'][0], extract_params(result_2)['c'][0])

plot_data(
    datasets=[
        {
            'xdata': data['1']['t'][char_mask],
            'ydata': data['1']['U'][char_mask],
            'xerr': 0.1,
            'yerr': 0.05,
            'fit': extract_params(result_1)['a'][0] *(1 - np.exp(-extract_params(result_1)['b'][0] *  np.linspace(0, 8.8, 100) + extract_params(result_1)['c'][0])), 
            'label': 'Aufladung',
            'fit_xdata': np.linspace(0, 8.8, 100),
            #'line': '-',
            'marker': '.',
        },
        {
            'xdata': data['1']['t'][dischar_mask],
            'ydata': data['1']['U'][dischar_mask],
            'xerr': 0.1,
            'yerr': 0.05,
            'fit': extract_params(result_2)['a'][0] * np.exp(extract_params(result_2)['b'][0] * ( np.linspace(0, 11.2, 100))) + extract_params(result_2)['c'][0],
            'label': 'Entladung',
            'fit_xdata': np.linspace(0, 11.2, 100)+8.8,
            #'line': '-',
            'marker': '.',
        },

    ],
    x_label='Zeit $t$/ ms',
    y_label='Spannung $U$/V',
    title='Auf und Entladung Kondensator',
    filename='Plots/WSK_1.pdf',
    width=25,
    height=20,
    plot=False,
)

datasets = []

for mask in [char_mask, dischar_mask]:
    if mask is char_mask:
        U_norm = 1 - data['1']['U'][mask] / max(data['1']['U'][mask])
        t = data['1']['t'][mask]
    else:
        U_norm = data['1']['U'][mask] / max(data['1']['U'][mask])
        t = data['1']['t'][mask] -8.8

    zero_mask = U_norm != 0
    U_norm = U_norm[zero_mask]
    U = data['1']['U'][mask][zero_mask]
    t = t[zero_mask]
    U_norm = np.log(U_norm)

    # sort t, U_norm, and U together
    sorted_indices = np.argsort(t)
    t = t[sorted_indices]
    U_norm = U_norm[sorted_indices]
    U = U[sorted_indices]



    if mask is char_mask:
        yerr = (
            np.log(np.clip(1 - (U + 0.05)/max(data['1']['U'][mask]), 1e-3, None)) - U_norm, 
            np.log(np.clip(1 - (U - 0.05)/max(data['1']['U'][mask]), 1e-3, None)) - U_norm
            )
        result = linear_fit(t, U_norm, 0.05 / (U - max(data['1']['U'][mask])), model='linear')
    else:
        yerr = (
            np.log(np.clip((U - 0.05)/U, 1e-2, None)), 
            np.log(np.clip((U + 0.05)/U, 1e-2, None)))
        result = linear_fit(t, U_norm, 0.05 / U, model='linear')

    datasets.append({
        'xdata': t,
        'ydata': U_norm,
        'yerr': yerr,
        'xerr': 0.1,
        'fit': extract_params(result)['a'][0] + extract_params(result)['b'][0] * t,
        'confidence': calc_CI(result, t),
        'label': 'Aufladung' if mask is char_mask else 'Entladung',
        'marker': '.',
    })

    print(f"{'Aufladung' if mask is char_mask else 'Entladung'}: ${print_round_val(extract_params(result)['b'][0], extract_params(result)['b'][1])} $")

plot_data(
    datasets=datasets,
    x_label='Zeit $t$/ms',
    y_label='ln$(1 - U/U_{max})$ bzw. ln$(U/U_{max})$',
    title='Auf und Entladung Kondensator',
    filename='Plots/WSK_2.pdf',
    width=25,
    height=20,
    plot=False,
)


print('-' * 100)

datasets_2 = []
for key in data['2']:
    if isinstance(data['2'][key], dict):
        xdata = np.log(data['2'][key]['f'])/np.log(10)
        ydata = 20*np.log(data['2'][key]['U']/data['2'][key]['U_0'])/np.log(10)

        dy = 20/np.log(10) * np.sqrt((data['2'][key]['dU']/data['2'][key]['U'])**2 + (data['2'][key]['dU_0']/data['2'][key]['U_0'])**2)

        print(key)

        result = None
        result2 = None
        if key == 'HP':
            result = linear_fit(xdata[:8], ydata[:8], dy[:8], model='linear')
            result2 = max(ydata) + 0.5
            f_th = 1/(2*np.pi*data['2'][key]['C']*data['2']['R'])

        elif key == 'TP':
            result = linear_fit(xdata[9:], ydata[9:], dy[9:], model='linear')
            result2 = np.mean(ydata[:4])
            f_th = (data['2']['R'] + data['2'][key]['R_L'])/(2*np.pi*data['2'][key]['L'])

        elif key == 'BP':
            result = linear_fit(xdata, ydata, dy, model='gaussian')

        if result2 is not None:
            a = extract_params(result)['a'][0]
            b = extract_params(result)['b'][0]
            da = extract_params(result)['a'][1]
            db = extract_params(result)['b'][1]

            print(f'm={b}')
            f_c = 10**((result2 - a) / b)
            df_c = f_c/np.log(10) * abs((result2 - a) / b)* np.sqrt((da / (result2 - a))**2 + (db / b)**2)
            print(f'f_c={f_c}')
            print(f'+{10**((result2 - a - da) / (b - db)) - f_c}')
            print(f' {10**((result2 - a + da) / (b + db)) - f_c}')
            print(f'\\pm {df_c}')
            print(f'f_th={f_th}')

            datasets_2.append({
                'xdata': xdata,
                'ydata': ydata,
                'yerr': dy,
                'fit': (extract_params(result)['a'][0] + extract_params(result)['b'][0] * xdata) if result else None,
                'fit_error_lines': [(
                    a + da + (b - db) * xdata,
                    a - da + (b + db) * xdata
                )] if result else None,
                'label': key,
                'line': 'None',
                'marker': '.',
                'color_group': key
            })

            datasets_2.append({
                'xdata': xdata[-12:] if key == 'HP' else xdata[:12],
                'ydata': [result2] * len(xdata[:12]),
                'line': '-',
                'marker': None,
                'label': None,
                'color_group': key
            })

            datasets_2.append({
                'xdata': np.log(f_th)/np.log(10),
                'ydata': result2 - 10*np.log(2)/np.log(10),
                'marker': "^",
                'label': "Theorie",
            })

        else:
            a, b, c, d = extract_params(result)['a'][0], extract_params(result)['b'][0], extract_params(result)['c'][0], extract_params(result)['d'][0]
            f_0 = b
            y_max = a + d
            f_L = b + c*np.sqrt(-2*np.log(1 -10*np.log(2)/np.log(10)/a))
            f_H = b - c*np.sqrt(-2*np.log(1 -10*np.log(2)/np.log(10)/a))


            f_0_th = 1/(2*np.pi*np.sqrt(data['2'][key]['L']*data['2'][key]['C']))
            f_L_th = (data['2'][key]['R_L'] + data['2']['R'])/(4*np.pi*data['2'][key]['L']) * (1 - np.sqrt(1 - 4*data['2'][key]['L']/data['2'][key]['C']/(data['2'][key]['R_L'] + data['2']['R'])**2))
            f_H_th = (data['2'][key]['R_L'] + data['2']['R'])/(4*np.pi*data['2'][key]['L']) * (1 + np.sqrt(1 - 4*data['2'][key]['L']/data['2'][key]['C']/(data['2'][key]['R_L'] + data['2']['R'])**2))

            print(f'f_0={10**f_0}')
            print(f'f_L={10**f_L}')
            print(f'f_H={10**f_H}')
            print(f'f_0_th={f_0_th}')
            print(f'f_L_th={f_L_th}')
            print(f'f_H_th={f_H_th}')

            datasets_2.append({
                'xdata': xdata,
                'ydata': ydata,
                'yerr': dy,
                'fit': a * np.exp(-((np.linspace(xdata.min(), xdata.max(), 300) - b) / c)**2/2) + d,
                'fit_error_lines': None,
                'fit_xdata': np.linspace(xdata.min(), xdata.max(), 300),
                'label': key,
                'line': 'None',
                'marker': '.',
                'color_group': key
            })
            datasets_2.append({
                'xdata':[f_L, f_H],
                'ydata': [y_max - 10*np.log(2)/np.log(10), y_max - 10*np.log(2)/np.log(10)],
                'marker': "^",
                'label': "BP Fit",
            })
            datasets_2.append({
                'xdata':[np.log(f_L_th)/np.log(10), np.log(f_H_th)/np.log(10)],
                'ydata': [y_max - 10*np.log(2)/np.log(10), y_max - 10*np.log(2)/np.log(10)],
                'marker': "^",
                'label': "Theorie",
            })

        print('-' * 100)


plot_data(
    datasets=[datasets_2[0], datasets_2[1], datasets_2[2]],
    x_label='Frequenz $log_{10}(f)/log_{10}(Hz)$',
    y_label='Spannung $20log_{10}(U)$/db',
    title='Hochpass',
    filename='Plots/WSK_HP.pdf',
    width=25,
    height=20,
    plot=False,
)

plot_data(
    datasets=[datasets_2[3], datasets_2[4], datasets_2[5]],
    x_label='Frequenz $log_{10}(f)/log_{10}(Hz)$',
    y_label='Spannung $20log_{10}(U)$/db',
    title='Tiefpass',
    filename='Plots/WSK_TP.pdf',
    ymax = 0,
    width=25,
    height=20,
    plot=True,
)

plot_data(
    datasets=[datasets_2[-3], datasets_2[-2], datasets_2[-1]],
    x_label='Frequenz $log_{10}(f)/log_{10}(Hz)$',
    y_label='Spannung $20log_{10}(U)$/db',
    title='Bandpass',
    filename='Plots/WSK_BP.pdf',
    width=25,
    height=20,
    plot=False,
)

