
import numpy as np
from plotting import plot_data
from tables import *
from help import *

data = {
    '1': {
        't': np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19.6, 11.2, 0.7, 1.6, 9.4, 3.6, 8.9, 8.8]),
        'U': np.array([0.5, 1.7, 2.05, 2.15, 2.18, 1.05, 0.30, 0.10, 0.05, 0, 0, 0.5, 1.05, 1.5, 1.5, 2, 2, 2.18]),
        'R': 17.81e3,
        'C': 0.1057e-6,
    },
    '2': {
        'R_{L_T}': 4,
        'R_{L_B}': 0.65,
        'dR_{L_B}': 0.05,
        'L_T': 4.720e-3,
        'dL_T': 0.05e-3,
        'L_B': 4.839e-6,
        'R': 8.5,
        'C_H': 3.354e-6,
        'C_B': 42.49e-6,
        'HP': {
            'f': np.array([10.04, 25.84, 40.10, 50.11, 100.15, 250.0, 500, 1.0015e3, 1.508e3, 2.0077e3, 3.0001e3, 3.5015e3, 5.0503e3, 7.5020e3, 10.093e3, 15.083e3, 25.05e3]),
            'U_0': np.array([0.79e3, 0.804e3, 0.805e3, 0.804e3, 0.799e3, 0.768e3, 0.681e3, 499.3, 376.7, 299.8, 213, 186, 134.4, 94.1, 71.3, 47.6, 25.2]),
            'U': np.array([1.5, 3.8, 5.8, 7.3, 14.4, 34.2, 60.4, 87.4, 97.1, 100, 99, 96.9, 88, 73.3, 60.4, 43, 23.5]),
            'df': np.array([0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.05, 0.001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.0001e3, 0.1e3])
        },
        'TP': {
            'f': np.array([10.01, 25.07, 40.19, 50.01, 100.15, 250.05, 405.55, 505.6, 1007.4, 1.504e3, 2.009e3, 3.002e3, 3.503e3, 5.007e3, 7.507e3, 10.073e3, 15.038e3, 24.985e3, 50.02]),
            'U_0': np.array([155, 158.9, 159.6, 160, 163.2, 183.1, 215.8, 240.1, 366.2, 466.7, 536.2, 602.1, 611.1, 593, 514.4, 434.3, 320.6, 199.8, 89.6]),
            'U': np.array([103, 105.7, 105.8, 106, 105.8, 105.2, 104.2, 103.1, 95.1, 84.9, 74.4, 56.8, 49.7, 34.2, 20.1, 12.4, 4.4, 0.3, 0.2]),
        },
        'BP': {
            'f': np.array([10.035, 25, 40.075, 50.015, 100.05, 249.9, 500, 1.005e3, 1.507e3, 2.004e3, 3.002e3, 3.506e3, 5.019e3, 7.498e3, 10.076e3, 15.027e3, 25.08e3, 749.3, 1.2508e3]),
            'U_0': np.array([0.775e3, 0.72e3, 0.642e3, 0.591e3, 403, 213.5, 147.9, 126.4, 127.1, 133, 149.6, 158.1, 180.4, 201.5, 207.5, 196.4, 153.8, 131.5, 125.7]),
            'U': np.array([20.6, 46.2, 64.4, 73.1, 95.4, 107.8, 110.3, 110.1, 108.5, 106.3, 100.6, 97.2, 86.0, 68.6, 53.9, 34.4, 13.6, 110.3, 109.3])
        }
    }
}

char_mask = data['1']['t'] <= 8.8
dischar_mask = data['1']['t'] >= 8.8

plot_data(
    datasets=[
        {
            'xdata': data['1']['t'][char_mask],
            'ydata': data['1']['U'][char_mask],
            'xerr': 0.1,
            'yerr': 0.05,
            'label': 'Aufladung',
            #'line': '-',
            'marker': '.',
        },
        {
            'xdata': data['1']['t'][dischar_mask],
            'ydata': data['1']['U'][dischar_mask],
            'xerr': 0.1,
            'yerr': 0.05,
            'label': 'Entladung',
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





def yerr_char(U, dU):
    return np.log(np.clip(1 - (U + dU) / max(U), 1e-3, None)) - np.log(np.clip(1 - U / max(U), 1e-3, None))

def yerr_dischar(U, dU):
    return np.log(np.clip((U + dU) / max(U), 1e-3, None)) - np.log(np.clip(U / max(U), 1e-3, None))


print( data['1']['t'][char_mask])
print(np.log(1 - data['1']['U'][char_mask] / max(data['1']['U'][char_mask])) )
print((yerr_char(data['1']['U'][char_mask], 0.05) + yerr_char(data['1']['U'][char_mask], -0.05)) / 2)




result_char = linear_fit(
    data['1']['t'][char_mask], 
    np.log(1 - data['1']['U'][char_mask] / max(data['1']['U'][char_mask])), 
    (yerr_char(data['1']['U'][char_mask], 0.05) + yerr_char(data['1']['U'][char_mask], -0.05)) / 2, 
    model='linear'
)

result_dischar = linear_fit(
    data['1']['t'][dischar_mask]-8.8,
    np.log(data['1']['U'][dischar_mask] / max(data['1']['U'][dischar_mask])),
    (yerr_dischar(data['1']['U'][dischar_mask], 0.05) + yerr_dischar(data['1']['U'][dischar_mask], -0.05)) / 2,
    model='linear'
)



char_data = {
    'xdata': data['1']['t'][char_mask],
    'ydata': np.log(1 - data['1']['U'][char_mask]/max(data['1']['U'][char_mask])),
    'yerr': (yerr_char(data['1']['U'][char_mask], -0.05), yerr_char(data['1']['U'][char_mask], 0.05)),
    'xerr': 0.1,
    'fit': extract_params(result_char)['a'][0] + extract_params(result_char)['b'][0] * data['1']['t'][char_mask],
    'label': 'Aufladung',
    'marker': '.',
}

dischar_data = {
    'xdata': data['1']['t'][dischar_mask]-8.8,
    'ydata': np.log(data['1']['U'][dischar_mask]/max(data['1']['U'][dischar_mask])),
    'yerr': (yerr_dischar(data['1']['U'][dischar_mask], -0.05), yerr_dischar(data['1']['U'][dischar_mask], 0.05)),
    'xerr': 0.1,
    'fit': extract_params(result_dischar)['a'][0] + extract_params(result_dischar)['b'][0] * (data['1']['t'][dischar_mask]-8.8),
    'label': 'Entladung',
    'marker': '.',
}

plot_data(
    datasets=[char_data, dischar_data],
    x_label='Zeit $t$/ms',
    y_label='ln$(1 - U/U_{max})$ bzw. ln$(U/U_{max})$',
    title='Auf und Entladung Kondensator',
    filename='Plots/WSK_2.pdf',
    width=25,
    height=20,
    plot=False,
)

datasets_2 = []
for key in data['2']:
    if isinstance(data['2'][key], dict):
        datasets_2.append({
            'xdata': np.log(data['2'][key]['f'])/np.log(10),
            'ydata': 20*np.log(data['2'][key]['U']/data['2'][key]['U_0'])/np.log(10),
            #'y_error': data['2'][key]['df'],
            'label': key,
            'line': 'None',
            'marker': '.',
        })

plot_data(
    datasets=[datasets_2[1]],
    x_label='Frequenz $log_{10}(f)/log_{10}(Hz)$',
    y_label='Spannung $log_{10}(U)$/db',
    title='Frequenzgang',
    filename='Plots/WSK_3.pdf',
    width=25,
    height=20,
    plot=True,
)