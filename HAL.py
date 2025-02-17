from unittest import result
import numpy as np
from plotting import plot_data, plot_color_seeds
from tables import *
from help import *


data = {
    'dI_S': [0.0005], 'dT': [0.5], 'd': 20e-3,
    '1':{
        'a':{
            'U_H': ['7.670', '8.929', '10.083', '11.199', '12.427', '13.674', '15.059', '16.274', '17.613', '18.988', '20.167', '21.450', '22.805', '24.020', '25.270'], 'U_H_0': ['-1.280'],
            'I_B': ['0.498', '0.584', '0.657', '0.725', '0.800', '0.874', '0.954', '1.026', '1.103', '1.183', '1.251', '1.326', '1.405', '1.476', '1.550'], 'dU_H' : [], 'dI_B' : [], 'dU_H_0': [],
            'I_S': ['0.02']
        },
        'b':{
            'U_H_0': ['-1.268', '-1.335', '-1.366', '-1.410', '-1.466', '-1.555', '-1.595', '-1.662', '-1.711', '-1.744', '-1.791', '-1.831', '-1.877', '-1.936', '-2.013', '-2.089'],
            'I_S': ['0.02', '0.019', '0.018', '0.017', '0.016', '0.014', '0.013', '0.011', '0.01', '0.009', '0.008', '0.007', '0.006', '0.005', '0.003', '0.002'],
            'U_H': ['25.234', '24.320', '22.484', '21.360', '19.652', '16.840', '15.388', '12.415', '10.714', '9.164', '7.232', '6.930', '5.066', '3.737', '0.750', '-1.105'],
            'I_B': ['1.529'], 'dU_H' : [], 'dI_B' : [], 'dU_H_0': [],
        },
    },

    '2':{
        'T':[],
        'U_H':[],
    }
}


def calc_dU(U):
    return np.array(U, dtype=float) * 0.004 + 1* last_digit(U)

def calc_dI_B(I_B):
    return np.array(I_B, dtype=float) * 0.008 + 5* last_digit(I_B)

for key in data['1']:
    data['1'][key]['U_H_0'] = np.array([Decimal(U) for U in data['1'][key]['U_H_0']])
    data['1'][key]['U_H'] = np.array([Decimal(U) for U in data['1'][key]['U_H']])
    data['1'][key]['I_B'] = np.array([Decimal(I) for I in data['1'][key]['I_B']])
    data['1'][key]['I_S'] = np.array([Decimal(I) for I in data['1'][key]['I_S']])

    data['1'][key]['dU_H'] = calc_dU(data['1'][key]['U_H'])
    data['1'][key]['dU_H_0'] = calc_dU(data['1'][key]['U_H_0'])
    data['1'][key]['dI_B'] = calc_dI_B(data['1'][key]['I_B'])

    data['1'][key]['U_H'] = data['1'][key]['U_H'] - data['1'][key]['U_H_0']

    data['1'][key]['U_H_0'] = np.array(data['1'][key]['U_H_0'], dtype=float)
    data['1'][key]['U_H'] = np.array(data['1'][key]['U_H'], dtype=float)
    data['1'][key]['I_B'] = np.array(data['1'][key]['I_B'], dtype=float)
    data['1'][key]['I_S'] = np.array(data['1'][key]['I_S'], dtype=float)


result_1_U = lmfit(data['1']['a']['I_B'], data['1']['a']['U_H'], data['1']['a']['dU_H'])
result_2_U = lmfit(data['1']['b']['I_S'], data['1']['b']['U_H'], data['1']['b']['dU_H'])
result_1_I = lmfit(data['1']['a']['U_H'], data['1']['a']['I_B'], data['1']['a']['dI_B'])
result_2_I = lmfit(data['1']['b']['U_H'], data['1']['b']['I_S'], data['dI_S']*np.ones(len(data['1']['b']['I_S'])))

print(result_1_U.fit_report())
print(result_2_U.fit_report())
print(result_1_I.fit_report())
print(result_2_I.fit_report())


plot_data(
    datasets= [
        {
            'xdata': data['1']['a']['I_B'],
            'ydata': data['1']['a']['U_H'],
            'xerr': data['1']['a']['dI_B'],
            'yerr': data['1']['a']['dU_H'],
        }
    ],
    x_label=r'$I_B \ [A]$',
    y_label=r'$U_H \ [mV]$',
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_1.pdf',
    plot=False
)

plot_data(
    datasets= [
        {
            'xdata': data['1']['b']['I_S'],
            'ydata': data['1']['b']['U_H'],
            'xerr': data['dI_S'],
            'yerr': data['1']['b']['dU_H'],
        }
    ],
    x_label=r'$I_B \ [A]$',
    y_label=r'$U_H \ [mV]$',
    xmin=0,
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_2.pdf',
    plot=True
)



