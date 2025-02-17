import numpy as np
from plotting import plot_data, plot_color_seeds
from tables import *
from help import *


data = {
    'dI_S': [0.0005], 'dT': [0.5], 'd': 20e-3,
    '1':{
        'a':{
            'U_H': ['7.670', '8.929', '10.083', '11.199', '12.427', '13.674', '15.059', '16.274', '17.613', '18.988', '20.167', '21.450', '22.805', '24.020', '25.270'], 'U_H_0': ['-1.280'],
            'I_B': ['0.498', '0.584', '0.657', '0.725', '0.800', '0.874', '0.954', '1.026', '1.103', '1.183', '1.251', '1.326', '1.405', '1.476', '1.550'], 'dU_H' : [], 'dI_B' : [],
            'I_S': ['20'], 'B': [], 'dB': [],
        },
        'b':{
            'U_H_0': ['-1.268', '-1.335', '-1.366', '-1.410', '-1.466', '-1.555', '-1.595', '-1.662', '-1.711', '-1.744', '-1.791', '-1.831', '-1.877', '-1.936', '-2.013', '-2.089'],
            'I_S': ['20', '19', '18', '17', '16', '14', '13', '11', '10', '9', '8', '7', '6', '5', '3', '2'],
            'U_H': ['25.234', '24.320', '22.484', '21.360', '19.652', '16.840', '15.388', '12.415', '10.714', '9.164', '7.232', '6.930', '5.066', '3.737', '0.750', '-1.105'],
            'I_B': ['1.529'], 'dU_H' : [], 'dI_B' : [], 'dU_H_0': [], 'B': [], 'dB': [],
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

def calc_B(I):
    return I*0.13636 + 0.004546

for key in data['1']:
    U_0 = np.array([Decimal(U) for U in data['1'][key]['U_H_0']])
    U_H = np.array([Decimal(U) for U in data['1'][key]['U_H']])
    I_B = np.array([Decimal(I) for I in data['1'][key]['I_B']])
    I_S = np.array([Decimal(I) for I in data['1'][key]['I_S']])

    data['1'][key]['dU_H'] = np.sqrt(calc_dU(U_H)**2 + calc_dU(U_0)**2)/1000
    data['1'][key]['dI_B'] = calc_dI_B(I_B)

    U_H = U_H - U_0

    data['1'][key]['U_H'] = np.array(U_H/1000, dtype=float)
    data['1'][key]['I_B'] = np.array(I_B, dtype=float)
    data['1'][key]['I_S'] = np.array(I_S/1000, dtype=float)

    data['1'][key]['B'] = calc_B(data['1'][key]['I_B'])
    data['1'][key]['dB'] = 0.13636 * data['1'][key]['dI_B']


result_1_I = lmfit(data['1']['a']['U_H'], data['1']['a']['B'], data['1']['a']['dB'])
result_2_I = lmfit(data['1']['b']['U_H'], data['1']['b']['I_S'], data['dI_S']*np.ones(len(data['1']['b']['I_S'])))

a1, da1, b1, db1, _, _ = slope(data['1']['a']['U_H'], data['1']['a']['B'], data['1']['a']['dB'])
a2, da2, b2, db2, _, _ = slope(data['1']['b']['U_H'], data['1']['b']['I_S'], data['dI_S']*np.ones(len(data['1']['b']['I_S'])))

print(result_1_I.fit_report())
print(result_2_I.fit_report())


x = np.linspace(data['1']['a']['U_H'][0], data['1']['a']['U_H'][-1], 100)
plot_data(
    datasets= [
        {
            'ydata': data['1']['a']['B'],
            'xdata': data['1']['a']['U_H'],
            'yerr': data['1']['a']['dB'],
            'xerr': data['1']['a']['dU_H'],
            'fit_xdata' :x,
            'fit': result_1_I.eval(x=x),
            'confidence': calc_CI(result_1_I, x)
        }
    ],
    y_label=r'$B \ [T]$',
    x_label=r'$U_H \ [V]$',
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_1.pdf',
    plot=False
)

x = np.linspace(data['1']['b']['U_H'][0], data['1']['b']['U_H'][-1], 100)

plot_data(
    datasets= [
        {
            'ydata': data['1']['b']['I_S'],
            'xdata': data['1']['b']['U_H'],
            'yerr': data['dI_S'],
            'xerr': data['1']['b']['dU_H'],
            'fit_xdata' :x,
            'fit': result_2_I.eval(x=x),
            'confidence': calc_CI(result_2_I, x)
        }
    ],
    y_label=r'$I_S \ [A]$',
    x_label=r'$U_H \ [V]$',
    ymin=0,
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_2.pdf',
    plot=False
)

B = data['1']['b']['B']

R_H_a = data['d']/ result_1_I.params['b'].value/ data['1']['a']['I_S'][0]
R_H_b = data['d']/ result_2_I.params['b'].value/ B[0]

dR_H_a = R_H_a * np.sqrt( (result_1_I.params['b'].stderr/result_1_I.params['b'].value)**2 + (data['dI_S'][0]/data['1']['a']['I_S'][0])**2)
dR_H_b = R_H_b * np.sqrt( (result_2_I.params['b'].stderr/result_2_I.params['b'].value)**2 + (data['1']['b']['dB'][0]/B[0])**2)

print(f'R_H_a = {round_val(R_H_a, dR_H_a)[0]} \\pm {round_val(R_H_a, dR_H_a)[1]}')
print(f'R_H_b = {round_val(R_H_b, dR_H_b)[0]} \\pm {round_val(R_H_b, dR_H_b)[1]}')

R_H_a_slope = data['d']/ b1/ data['1']['a']['I_S'][0]
R_H_b_slope = data['d']/ b2/ B[0]
dR_H_a_slope = R_H_a_slope * np.sqrt( (db1/b1)**2 + (data['dI_S'][0]/data['1']['a']['I_S'][0])**2)
dR_H_b_slope = R_H_b_slope * np.sqrt( (db2/b2)**2 + (data['1']['b']['dB'][0]/B[0])**2)

print(f'R_H_a_slope = {round_val(R_H_a_slope, dR_H_a_slope)[0]} \\pm {round_val(R_H_a_slope, dR_H_a_slope)[1]}')
print(f'R_H_b_slope = {round_val(R_H_b_slope, dR_H_b_slope)[0]} \\pm {round_val(R_H_b_slope, dR_H_b_slope)[1]}')

