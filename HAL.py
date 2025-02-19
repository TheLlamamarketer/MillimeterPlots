import numpy as np
from plotting import plot_data, plot_color_seeds
from tables import *
from help import *
from scipy.interpolate import UnivariateSpline
import warnings


data = {
    'dI_S': [0.0005], 'dT': [0.5], 'd': 1e-3,
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
        'T': [125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 
          105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 
          81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 
          56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 
          31, 30, 29, 28, 27],
        'U_H': ['5.0895', '5.2425', '5.4235', '5.621', '5.826', '6.031', '6.2125', '6.496', '6.774', '6.9535', '7.1795', 
                '7.455', '7.6665', '7.885', '8.2255', '8.57', '8.837', '9.0955', '9.4545', '9.7355', '10.022', '10.432', 
                '10.75', '11.112', '11.4385', '11.772', '12.1165', '12.509', '12.936', '13.287', '13.684', '14.0925', 
                '14.496', '14.869', '15.2805', '15.696', '16.0545', '16.4195', '16.7905', '17.217', '17.624', '18.0095', 
                '18.3325', '18.7645', '19.089', '19.458', '19.762', '20.1255', '20.467', '20.7755', '21.0395', '21.3345', 
                '21.623', '21.8645', '22.1145', '22.344', '22.543', '22.724', '22.938', '23.1355', '23.312', '23.488', 
                '23.642', '23.782', '23.898', '24.0315', '24.154', '24.2455', '24.3645', '24.448', '24.5315', '24.6045', 
                '24.6735', '24.7345', '24.7975', '24.845', '24.893', '24.9345', '24.9705', '24.999', '25.0265', '25.05', 
                '25.0695', '25.0895', '25.1065', '25.118', '25.127', '25.133', '25.135', '25.138', '25.1345', '25.1325', 
                '25.1325', '25.13', '25.1275', '25.129', '25.1305', '25.142', '25.185'],
    }
}


def calc_dU(U):
    return np.array(U, dtype=float) * 0.004 + 5* last_digit(U)

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
    data['1'][key]['dB'] = np.sqrt((0.13636 * data['1'][key]['dI_B'])**2 + 0.005**2)


result_1_I = lmfit(data['1']['a']['U_H'], data['1']['a']['B'], data['1']['a']['dB'])
result_2_I = lmfit(data['1']['b']['U_H'], data['1']['b']['I_S'], data['dI_S']*np.ones(len(data['1']['b']['I_S'])))
result_1_U = lmfit(data['1']['a']['B'], data['1']['a']['U_H'], data['1']['a']['dU_H'])
result_2_U = lmfit(data['1']['b']['I_S'], data['1']['b']['U_H'], data['1']['b']['dU_H'])


a1, da1, b1, db1, _, _ = slope(data['1']['a']['U_H'], data['1']['a']['B'], data['1']['a']['dB'])
a2, da2, b2, db2, _, _ = slope(data['1']['b']['U_H'], data['1']['b']['I_S'], data['dI_S']*np.ones(len(data['1']['b']['I_S'])))

print(result_1_I.fit_report())
print(result_2_I.fit_report())
print(result_1_U.fit_report())
print(result_2_U.fit_report())



B = data['1']['b']['B']

R_H_a = data['d']/ result_1_I.params['b'].value/ data['1']['a']['I_S'][0]*10**3
R_H_b = data['d']/ result_2_I.params['b'].value/ B[0]*10**3

dR_H_a = R_H_a * np.sqrt( (result_1_I.params['b'].stderr/result_1_I.params['b'].value)**2 + (data['dI_S'][0]/data['1']['a']['I_S'][0])**2)
dR_H_b = R_H_b * np.sqrt( (result_2_I.params['b'].stderr/result_2_I.params['b'].value)**2 + (data['1']['b']['dB'][0]/B[0])**2)

print(f'R_H_a = ({round_val(R_H_a, dR_H_a)[0]} \\pm {round_val(R_H_a, dR_H_a)[1]}) 10^{{-3}} \\ m^3/C')
print(f'R_H_b = ({round_val(R_H_b, dR_H_b)[0]} \\pm {round_val(R_H_b, dR_H_b)[1]}) 10^{{-3}} \\ m^3/C')

R_H_a_slope = data['d']/ b1/ data['1']['a']['I_S'][0]*10**3
R_H_b_slope = data['d']/ b2/ B[0]*10**3
dR_H_a_slope = R_H_a_slope * np.sqrt( (db1/b1)**2 + (data['dI_S'][0]/data['1']['a']['I_S'][0])**2)
dR_H_b_slope = R_H_b_slope * np.sqrt( (db2/b2)**2 + (data['1']['b']['dB'][0]/B[0])**2)

print(f'R_H_a_slope = ({round_val(R_H_a_slope, dR_H_a_slope)[0]} \\pm {round_val(R_H_a_slope, dR_H_a_slope)[1]})10^{{-3}}m^3/C')
print(f'R_H_b_slope = ({round_val(R_H_b_slope, dR_H_b_slope)[0]} \\pm {round_val(R_H_b_slope, dR_H_b_slope)[1]})10^{{-3}}m^3/C')



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
            'confidence': calc_CI(result_1_I, x, 3)
        }
    ],
    y_label=r'$B \ [T]$',
    x_label=r'$U_H \ [V]$',
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_I1.pdf',
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
            'confidence': calc_CI(result_2_I, x, 3)
        }
    ],
    y_label=r'$I_S \ [A]$',
    x_label=r'$U_H \ [V]$',
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_I2.pdf',
    plot=False
)

U_H = np.array([Decimal(U) for U in data['2']['U_H']])
T = np.array([Decimal(T) for T in data['2']['T']])

data['2']['U_H'] = np.array(U_H/1000, dtype=float)
data['2']['T'] = np.array(T, dtype=float) + 273.15


y = 3 * np.log(data['2']['U_H']* data['2']['T'])
x = 1/data['2']['T']
dx = data['dT'] * np.ones(len(data['2']['T']))/data['2']['T']**2

spline = UnivariateSpline(x, y, s=0.005, k=4)
spline_der = spline.derivative()
spline_der2 = spline.derivative(n=2)

x_smooth = np.linspace(np.min(x), np.max(x), 300)

m_spline = spline_der(x[np.argmin(spline_der2(x))])

# index size between each point 0.0000085 1/K 

indexes = np.where((x >= 0.002650) & (x <= 0.002751))
indexes2 = np.where((x >= 0.00272) & (x <= 0.00278))
#indexes = np.where((x >= 0.0027) & (x <= 0.002751))

result = lmfit(x[indexes], y[indexes], None)
result2 = lmfit(x[indexes2], y[indexes2], None)
k_b = 1.38e-23
e = 1.602e-19

E_1 = round_val(result.params['b'].value*k_b/e, result.params['b'].stderr*k_b/e)
print(f"$E_1 = {E_1[0]} \\pm {E_1[1]}$")
E_2 = round_val(result2.params['b'].value*k_b/e, result2.params['b'].stderr*k_b/e)
print(f"$E_2 = {E_2[0]} \\pm {E_2[1]}$")
print(f"m_spline = {m_spline*k_b/e}")



def plot_intervals_r2(x, y, min_window_size=5, max_window_size=20):
    colors = plt.cm.viridis(np.linspace(0, 1, max_window_size - min_window_size + 1))
    datasets = []
    best_intervals = []

    for window_size, color in zip(range(min_window_size, max_window_size + 1), colors):
        intervals = []
        r2_values = []
        n_points = len(x)
        for start in range(n_points - window_size):
            end = start + window_size
            result = lmfit(x[start:end], y[start:end], None)
            r2 = result.rsquared
            intervals.append(x[start])
            r2_values.append(r2)
        
        best_index = np.argmax(r2_values)
        best_intervals.append((window_size, intervals[best_index], r2_values[best_index]))

        datasets.append({
            'ydata': r2_values,
            'xdata': intervals,
            'label': f'Window size {window_size}',
            'color': color
        })
    
    plot_data(
        datasets=datasets,
        y_label=r'$R^2$',
        x_label=r'Starting value of interval',
        ymin=0.99,
        ymax=1,
        title='R^2 values for different intervals',
        filename=f'Plots/R2_intervals.pdf',
        plot=False
    )

    # Evaluate the best intervals based on a score
    scores = [(window_size, start, r2, r2) for window_size, start, r2 in best_intervals]
    scores.sort(key=lambda x: x[3], reverse=True)

    for window_size, start, r2, score in scores:
        print(f'Window size: {window_size}, Start: {start:.6f}, R^2: {r2:.4f}, Score: {score:.4f}')

#plot_intervals_r2(x[:-50], y[:-50], min_window_size=5, max_window_size=18)


plot_data(
    datasets= [
        {
            'ydata': y,
            'xdata': x,
            'xerr': dx,
            'fit_xdata' :x_smooth,
            'fit': spline(x_smooth),
            'color_group':'1',
        },
        {
            'ydata': result.eval(x=x),
            'xdata': x,
            'line': '-',
            'marker': None,
            'label': 'Fit',
            'color_group':'1',
        },
        {
            'ydata': result2.eval(x=x),
            'xdata': x,
            'line': '-',
            'marker': None,
            'label': 'Fit',
            'color_group':'2',
        },
    ],
    y_label=r'$3 \ ln(U_H \cdot T) $',
    x_label=r'$1/T \ [1/K]$',
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_2.pdf',
    ymax = 6.4,
    plot=False
)

plot_data(
    datasets= [
        {
            'ydata': spline(x)-y,
            'xdata': x,
        }
    ],
    y_label=r'$3 \ ln(U_H \cdot T) - y$',
    x_label=r'$1/T \ [1/K]$',
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_2_res.pdf',
    plot=False
)

plot_data(
    datasets= [
        {
            'ydata': spline_der(x_smooth)*k_b/e,
            'xdata': x_smooth,
            'line': '-',
            'marker': None,
        }
    ],
    y_label=r'$\frac{d}{dT} 3 \ ln(U_H \cdot T) $',
    x_label=r'$1/T \ [1/K]$',
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_2_der.pdf',
    plot=False
)

plot_data(
    datasets= [
        {
            'ydata': spline_der2(x_smooth),
            'xdata': x_smooth,
            'line': '-',
            'marker': None,
        }
    ],
    y_label=r'$\frac{d^2}{dT^2} 3 \ ln(U_H \cdot T) $',
    x_label=r'$1/T \ [1/K]$',
    title='Hallspannung in Magnetfeld',
    filename=f'Plots/HAL_2_der2.pdf',
    plot=False
)
