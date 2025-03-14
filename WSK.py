import numpy as np
from plotting import plot_data, plot_color_seeds
from tables import *
from help import *
from decimal import Decimal

data = {
    '1': {
        't': np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19.6, 11.2, 0.7, 1.6, 9.4, 3.6, 8.9, 8.8]),
        'U': np.array([0.5, 1.7, 2.05, 2.15, 2.18, 1.05, 0.30, 0.10, 0.05, 0, 0, 0.5, 1.05, 1.5, 1.5, 2, 2, 2.18]),
        'R': 17.81e3,
        'dR': 0.1803e3,
        'C': 105.7e-9,
        'dC': 0.007*105.7e-9 + 0.3e-9,
    },
    '2': {
        'R': 8.5,
        'dR': 0.009*8.5 + 0.2,
        'HP': {
            'f': np.array([10.04, 25.84, 40.10, 50.11, 100.15, 250.0, 500, 1.0015e3, 1.508e3, 2.0077e3, 3.0001e3, 3.5015e3, 5.0503e3, 7.5020e3, 10.093e3, 15.083e3, 25.05e3]),
            'U_0': np.array([0.79e3, 0.804e3, 0.805e3, 0.804e3, 0.799e3, 0.768e3, 0.681e3, 499.3, 376.7, 299.8, 213, 186, 134.4, 94.1, 71.3, 47.6, 25.2]),
            'U': np.array([1.5, 3.8, 5.8, 7.3, 14.4, 34.2, 60.4, 87.4, 97.1, 100, 99, 96.9, 88, 73.3, 60.4, 43, 23.5]),
            'df': np.array([0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.05, 0.001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.1e3]),
            'dU': np.array([0.2023, 0.2057, 0.2087, 0.2109, 0.2216, 0.2513, 0.2906, 0.3311, 0.3457, 0.3500, 0.3485, 0.3454, 0.3320, 0.3100, 0.2906, 0.2645, 0.2353]),
            'dU_0': np.array([10, 3.2060, 3.2075, 3.2060, 3.1985, 3.1520, 3.0215, 0.9489, 0.7651, 0.6497, 2.3195, 2.2790, 0.4016, 0.3412, 0.3070, 0.2714, 0.2378]),
            'C': 3.354e-6, 'dC': 0.026478e-6,
        },
        'TP': {
            'f': np.array([10.01, 25.07, 40.19, 50.01, 100.15, 250.05, 405.55, 505.6, 1007.4, 1.504e3, 2.009e3, 3.002e3, 3.503e3, 5.007e3, 7.507e3, 10.073e3, 15.038e3, 24.985e3, 50.02e3]),
            'U_0': np.array([155, 158.9, 159.6, 160, 163.2, 183.1, 215.8, 240.1, 366.2, 466.7, 536.2, 602.1, 611.1, 593, 514.4, 434.3, 320.6, 199.8, 89.6]),
            'U': np.array([103, 105.7, 105.8, 106, 105.8, 105.2, 104.2, 103.1, 95.1, 84.9, 74.4, 56.8, 49.7, 34.2, 20.1, 12.4, 4.4, 1.3, 0.2]),
            'dU': np.array([1, 0.3586, 0.3587, 0.3587, 0.3587, 0.3578, 0.3563, 0.3547, 0.3427, 0.3274, 0.3116, 0.2852, 0.2746, 0.2513, 0.2302, 0.2186, 0.2066, 0.2020, 0.2003]),
            'dU_0': np.array([2, 0.4384, 0.4394, 0.4400, 0.4448, 0.4747, 0.5237, 0.5602, 0.7493, 0.9001, 1.0043, 1.1032, 1.1167, 2.8895, 0.9716, 0.8515, 0.6809, 0.4997, 0.3344]),
            'L': 4.720e-3, 'dL': 0.05e-3, 'R_L': 4.0, 'dR_L': 4*0.009 + 0.2,
        },
        'BP': {
            'f': np.array([10.035, 25, 40.075, 50.015, 100.05, 249.9, 500, 1.005e3, 1.507e3, 2.004e3, 3.002e3, 3.506e3, 5.019e3, 7.498e3, 10.076e3, 15.027e3, 25.08e3, 749.3, 1.2508e3]),
            'U_0': np.array([0.775e3, 0.72e3, 0.642e3, 0.591e3, 403, 213.5, 147.9, 126.4, 127.1, 133, 149.6, 158.1, 180.4, 201.5, 207.5, 196.4, 153.8, 131.5, 125.7]),
            'U': np.array([20.6, 46.2, 64.4, 73.1, 95.4, 107.8, 110.3, 110.1, 108.5, 106.3, 100.6, 97.2, 86.0, 68.6, 53.9, 34.4, 13.6, 110.3, 109.3]),
            'dU':np.array([0.2309, 0.2693, 0.2966, 0.3096, 0.3431, 0.3617, 0.3654, 0.3651, 0.3628, 0.3595, 0.3509, 0.3458, 0.3290, 0.3029, 0.2809, 0.2516, 0.2204, 0.3654, 0.3639]),
            'dU_0': np.array([5, 1.0800, 2.9630, 2.8865, 2.6045, 0.3202, 0.2219, 0.1896, 0.1906, 2.1995, 0.2244, 0.2371, 0.2706, 0.3023, 0.3113, 0.2946, 0.2307, 0.1973, 0.1885]),
            'C': 42.49e-6, 'dC': 42.49e-6 * 0.007 + 0.03e-6, 'L': 4.839e-6, 'R_L': 0.65, 'dR_L': 0.05,
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


header_groups = [('Aufladung', 2), ('Entladung', 2)]
headers = {'t_A': {'label': '{Zeit $t/ms$}', 'err': 0.1, 'data': data['1']['t'][char_mask]},
            'U_A': {'label': '{Spannung $U/V$}', 'err': 0.05, 'data': data['1']['U'][char_mask]},
            't_E': {'label': '{Zeit $t/ms$}', 'err': 0.1, 'data': data['1']['t'][dischar_mask]},
            'U_E': {'label': '{Spannung $U/V$}', 'err': 0.05, 'data': data['1']['U'][dischar_mask]}
}

print_standard_table(
    data=data['1'],
    headers=headers,
    header_groups=header_groups,
    column_formats= ["2.1"] * len(headers),
    caption="d",
    label="tab:A1",
    show=False
)

print('-' * 100)

print(f'$\\tau_{{th}}={print_round_val(10**3*data["1"]["R"]*data["1"]["C"], 10**3*np.sqrt((data["1"]["C"]*data["1"]["dR"])**2 + (data["1"]["R"]*data["1"]["dC"])**2))}ms$')

result_1 = lmfit(data['1']['t'][char_mask], data['1']['U'][char_mask], model='exponential_decay', constraints={"a": {"min": 0}, "b": {"min": 0}})
print(f'$\\tau_{{Auf}}={print_round_val(1/extract_params(result_1)["b"][0], extract_params(result_1)["b"][1]/extract_params(result_1)["b"][0]**2)} \\ ms$')

result_2 = lmfit(data['1']['t'][dischar_mask]-8.8, data['1']['U'][dischar_mask], model='exponential', constraints={"a": {"min": 0}, "c":0})
print(f'$\\tau_{{Ent}}={print_round_val(-1/extract_params(result_2)["b"][0], extract_params(result_2)["b"][1]/extract_params(result_2)["b"][0]**2)} \\ ms$')

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
    color_seed=39,
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
        result = lmfit(t, U_norm, 0.05 / (U - max(data['1']['U'][mask])), model='linear', const_weight=100)
    else:
        yerr = (
            np.log(np.clip((U - 0.05)/U, 1e-2, None)), 
            np.log(np.clip((U + 0.05)/U, 1e-2, None)))
        result = lmfit(t, U_norm, 0.05 / U, model='linear', const_weight=100)

    datasets.append({
        'xdata': t,
        'ydata': U_norm,
        'yerr': yerr,
        'xerr': 0.1,
        'fit': extract_params(result)['a'][0] + extract_params(result)['b'][0] * t,
        'confidence': calc_CI(result, t),
        'label': 'Aufladung' if mask is char_mask else 'Entladung',
    })

    print(f"${'\\tau_{Auf}' if mask is char_mask else '\\tau_{Ent}'}: {print_round_val(-1/extract_params(result)['b'][0], extract_params(result)['b'][1]/extract_params(result)['b'][0]**2)} \\ ms$")



plot_data(
    datasets=datasets,
    x_label='Zeit $t$/ms',
    y_label='ln$(1 - U/U_{max})$ bzw. ln$(U/U_{max})$',
    title='Auf und Entladung Kondensator',
    filename='Plots/WSK_2.pdf',
    ymin=-5,
    width=25,
    height=20,
    color_seed=39,
    plot=False,
)

def fit(y1, y2, x1, x2, dy1=None, dy2=None):
    y1_mean = np.mean(y1)
    y2_mean = np.mean(y2)
    y = np.concatenate([y1, y2])
    y_mean = np.mean(y)
    n1 = len(y1)
    n2 = len(y2)
    n = n1 + n2

    x1_mean = np.mean(x1)
    x2_mean = np.mean(x2)
    x = np.concatenate([x1, x2])
    x_mean = np.mean(x)

    weights = np.concatenate([1/dy1[0]**2, 1/dy2[0]**2]) if dy1 is not None and dy2 is not None else np.ones(len(y1) + len(y2))
    Sxx = np.sum(weights * (x - x_mean)**2)
    Sxy = np.sum(weights * (x - x_mean) * (y - y_mean))
    Sw = np.sum(weights)
    Syy = np.sum(weights * (y - y_mean)**2)

    b = Sxy/Sxx
    a1 = y1_mean - b * x1_mean
    a2 = y2_mean - b * x2_mean
    a = y_mean - b * x_mean

    r = Sxy / np.sqrt(Sxx * Syy)
    d = np.sqrt((1 - r**2) * Syy / (n - 2))
    da = d * np.sqrt(1/Sw + x_mean**2/Sxx)
    db = d / np.sqrt(Sxx)
    dab = -d * x_mean / Sxx


    return a1, a2, b, da, db, dab

def calc_CI_normal(fit, xdata, da, db, dab, sigmas=[3]):
    ci_list = []
    try:
        for sigma in sigmas:
            # Calculate uncertainty at the given sigma level
            uncertainty = sigma * np.sqrt(da**2 + (xdata * db)**2 + 2 * xdata * dab)
            lower = fit - uncertainty
            upper = fit + uncertainty

            ci_list.append((lower, upper))
    except Exception as e:
        import logging
        logging.error(f"Error calculating confidence intervals: {e}")
        for sigma in sigmas:
            ci_list.append((np.full_like(xdata, np.nan), np.full_like(xdata, np.nan)))
    return ci_list



y = tuple(dataset['ydata'] for dataset in datasets)
x = tuple(dataset['xdata'] for dataset in datasets)
dy = tuple(dataset['yerr'] for dataset in datasets)
a1, a2, b, da, db, dab = fit(*y, *x, *dy)

for dataset in datasets:
    fit = a1 + b * dataset['xdata'] if dataset['label'] == 'Aufladung' else a2 + b * dataset['xdata']
    dataset['fit'] = fit
    dataset['confidence'] = calc_CI_normal(fit, dataset['xdata'], da, db, dab)

plot_data(
    datasets=datasets,
    x_label='Zeit $t$/ms',
    y_label='ln$(1 - U/U_{max})$ bzw. ln$(U/U_{max})$',
    title='Auf und Entladung Kondensator',
    filename='Plots/WSK_2b.pdf',
    ymin=-5,
    width=25,
    height=20,
    color_seed=39,
    plot=False,
)




print('-' * 100)

datasets_2 = []
headers = {'HP_f':{}, 'HP_U':{}, 'TP_f':{}, 'TP_U':{}, 'BP_f':{}, 'BP_U':{}}

'''
'HP': {
    'f': np.array([10.04, 25.84, 40.10, 50.11, 100.15, 250.0, 500, 1.0015e3, 1.508e3, 2.0077e3, 3.0001e3, 3.5015e3, 5.0503e3, 7.5020e3, 10.093e3, 15.083e3, 25.05e3]),
    'U_0': np.array([0.79e3, 0.804e3, 0.805e3, 0.804e3, 0.799e3, 0.768e3, 0.681e3, 499.3, 376.7, 299.8, 213, 186, 134.4, 94.1, 71.3, 47.6, 25.2]),
    'U': np.array([1.5, 3.8, 5.8, 7.3, 14.4, 34.2, 60.4, 87.4, 97.1, 100, 99, 96.9, 88, 73.3, 60.4, 43, 23.5]),
    'df': np.array([0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.05, 0.001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.1e3]),
    'dU': np.array([0.2023, 0.2057, 0.2087, 0.2109, 0.2216, 0.2513, 0.2906, 0.3311, 0.3457, 0.3500, 0.3485, 0.3454, 0.3320, 0.3100, 0.2906, 0.2645, 0.2353]),
    'dU_0': np.array([10, 3.2060, 3.2075, 3.2060, 3.1985, 3.1520, 3.0215, 0.9489, 0.7651, 0.6497, 2.3195, 2.2790, 0.4016, 0.3412, 0.3070, 0.2714, 0.2378]),
    'C': 3.354e-6, 'dC': 0.026478e-6,
},
'''

for key in data['2']:
    if isinstance(data['2'][key], dict):
        U = data['2'][key]['U']
        dU = data['2'][key]['dU']
        U_0 = data['2'][key]['U_0']
        dU_0 = data['2'][key]['dU_0']

        headers[key + '_f'] = {'label': '{$f$/Hz}', 'data': data['2'][key]['f']}
        headers[key + '_U'] = {'label': '{$U/U_0$}', 'err': U/U_0 * np.sqrt((dU/U)**2 + (dU_0/U_0)**2), 'data': U/U_0}

        xdata = np.log(data['2'][key]['f'])/np.log(10)
        ydata = 20*np.log(data['2'][key]['U']/data['2'][key]['U_0'])/np.log(10)

        dy = 20/np.log(10) * np.sqrt((data['2'][key]['dU']/data['2'][key]['U'])**2 + (data['2'][key]['dU_0']/data['2'][key]['U_0'])**2)

        print(key)

        result = None
        result2 = None
        if key == 'HP':
            result = lmfit(xdata[:8], ydata[:8], dy[:8], model='linear')
            result2 = max(ydata) + 0.5
            f_th = 1/(2*np.pi*data['2'][key]['C']*data['2']['R'])
            df_th = f_th * np.sqrt((data['2'][key]['dC']/data['2'][key]['C'])**2 + (data['2']['dR']/data['2']['R'])**2)

        elif key == 'TP':
            result = lmfit(xdata[9:], ydata[9:], dy[9:], model='linear')
            result2 = np.mean(ydata[:4])
            f_th = (data['2']['R'] + data['2'][key]['R_L'])/(2*np.pi*data['2'][key]['L'])
            df_th = 1/(2*np.pi*data['2'][key]['L'])*np.sqrt(data['2']['dR']**2 + data['2'][key]['dR_L']**2 + (data['2'][key]['dL']*(data['2']['R'] + data['2'][key]['R_L']))**2)

        elif key == 'BP':
            result = lmfit(xdata, ydata, dy, model='gaussian')

        if result2 is not None:
            a, da, b, db = extract_params(result)['a'][0], extract_params(result)['a'][1], extract_params(result)['b'][0], extract_params(result)['b'][1]

            print(f'$m_{key}={print_round_val(b, db, False)}$')
            f_c = 10**((result2 - a) / b)
            df_c = f_c/np.log(10) * abs((result2 - a) / b)* np.sqrt((da / (result2 - a))**2 + (db / b)**2)
            print(f'f_{{eck}}={print_round_val(f_c, df_c)}Hz, mit Grenzgeraden: {print_round_val(f_c, (10**((result2 - a - da) / (b - db)) - 10**((result2 - a + da) / (b + db)))/2)}Hz')
            print(f'f_{{th}}={print_round_val(f_th, df_th)}Hz')

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
                'xerr': df_th/f_th/np.log(10),
                'marker': "^",
                'label': "Theorie",
            })

        else:
            a, b, c, d = extract_params(result)['a'][0], extract_params(result)['b'][0], extract_params(result)['c'][0], extract_params(result)['d'][0]
            da, db, dc, dd = extract_params(result)['a'][1], extract_params(result)['b'][1], extract_params(result)['c'][1], extract_params(result)['d'][1]
            f_0 = b
            y_max = a + d
            f_L = b + c*np.sqrt(-2*np.log(1 -10*np.log(2)/np.log(10)/a))
            f_H = b - c*np.sqrt(-2*np.log(1 -10*np.log(2)/np.log(10)/a))

            df_L = np.sqrt(db**2 + dc**2*(-2*np.log(1 -10*np.log(2)/np.log(10)/a)) + ((da * 5 * np.sqrt(2) * c * np.log(2)) / (a * (a * np.log(10) - 10 * np.log(2)) * np.sqrt(-np.log(1 - (10 * np.log(2)) / (a * np.log(10))))))**2)
            df_H = df_L

            f_0_th = 1/(2*np.pi*np.sqrt(data['2'][key]['L']*data['2'][key]['C']))
            f_L_th = (data['2'][key]['R_L'] + data['2']['R'])/(4*np.pi*data['2'][key]['L']) * (1 - np.sqrt(1 - 4*data['2'][key]['L']/data['2'][key]['C']/(data['2'][key]['R_L'] + data['2']['R'])**2))
            f_H_th = (data['2'][key]['R_L'] + data['2']['R'])/(4*np.pi*data['2'][key]['L']) * (1 + np.sqrt(1 - 4*data['2'][key]['L']/data['2'][key]['C']/(data['2'][key]['R_L'] + data['2']['R'])**2))

            print(f'f_0={print_round_val(10**f_0, 10**f_0*np.log(10) * db)}Hz, f_0_th={print_round_val(f_0_th)}Hz')
            print(f'f_L={print_round_val(10**f_L, 10**f_L*np.log(10) * df_L)}Hz')
            print(f'f_H={print_round_val(10**f_H, 10**f_H*np.log(10) * df_H)}Hz')

            datasets_2.append({
                'xdata': xdata,
                'ydata': ydata,
                'yerr': dy,
                'fit': a * np.exp(-((np.linspace(xdata.min(), xdata.max(), 300) - b) / c)**2/2) + d,
                'fit_xdata': np.linspace(xdata.min(), xdata.max(), 300),
                'confidence': calc_CI(result, np.linspace(xdata.min(), xdata.max(), 300)),
                'label': key,
                'line': 'None',
                'marker': '.',
                'color_group': key
            })
            datasets_2.append({
                'xdata':[f_L, f_H, f_0],
                'ydata': [y_max - 10*np.log(2)/np.log(10), y_max - 10*np.log(2)/np.log(10), y_max],
                'xerr': [df_L, df_H, db],
                'marker': "^",
                'label': "BP Fit",
            })
        print('-' * 100)


header_groups = [('Hochpass', 2), ('Tiefpass', 2), ('Bandpass', 2)]



print_standard_table(
    data=data['2'],
    headers=headers,
    header_groups=header_groups,
    column_formats= ["2.1"] * len(headers),
    caption="Kennlinien für die verschiedenen Materialien. Die Spannung $U$ und Stromstärke $I$ wurden noch nicht auf die maximalen Werte normiert.",
    label="tab:A2",
    show=False
)


plot_data(
    datasets=[datasets_2[0], datasets_2[1], datasets_2[2]],
    x_label='Frequenz $log_{10}(f)/log_{10}(Hz)$',
    y_label='Spannung $20log_{10}(U)$/db',
    title='Hochpass',
    filename='Plots/WSK_HP.pdf',
    ymax=1,
    width=25,
    height=20,
    color_seed=39,
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
    color_seed=39,
    plot=False,
)

plot_data(
    datasets=[datasets_2[-2], datasets_2[-1]],
    x_label='Frequenz $log_{10}(f)/log_{10}(Hz)$',
    y_label='Spannung $20log_{10}(U)$/db',
    title='Bandpass',
    filename='Plots/WSK_BP.pdf',
    width=25,
    height=20,
    color_seed=39,
    plot=False,
)

#plot_color_seeds((0,50), 2)
