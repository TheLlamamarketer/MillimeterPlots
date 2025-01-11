from random import seed
import numpy as np
from plotting_minus import plot_data
from tables import *
from help import *
from scipy.interpolate import UnivariateSpline
from decimal import Decimal, getcontext

getcontext().prec = 20

print('-'*100)

data = {
    '1': {
        'R': {
            'U': [
                "0.000", "2.003", "3.998", "6.008", "8.00", "10.03", "12.02", "14.00",
                "16.00", "18.00", "20.00", "22.00", "24.00", "26.00", "28.00", "30.00", "30.99"
            ],
            'I': [
                "0", "4.27", "8.52", "12.80", "17.05", "21.37", "25.63", "29.85",
                "34.15", "38.47", "42.8", "47.2", "51.6", "56.2", "60.6", "65.0", "67.2"
            ],
            'R': ["468.2"],
            'dU': [],
            'dI': [],
            'dR': []
        },
        'Lamp': {
            'U': [
                "0", "0.500", "1.000", "1.500", "2.000", "2.503", "3.001", "3.500",
                "4.000", "4.500", "5.000", "5.52", "6.00"
            ],
            'I': [
                "0", "36.89", "51.3", "63.7", "74.6", "84.6", "93.9", "102.6",
                "110.8", "118.6", "126.1", "133.0", "140.3"
            ],
            'dU': [],
            'dI': []
        },
        'Graphite': {
            'U': [
                "0", "0.863", "1.713", "2.558", "3.381", "4.181", "4.973", "5.669",
                "6.401", "7.10", "7.74", "8.39", "8.92", "9.49", "9.73"
            ],
            'I': [
                "0", "0.075", "0.150", "0.225", "0.300", "0.375", "0.450", "0.525",
                "0.600", "0.675", "0.750", "0.825", "0.900", "0.975", "1.000"
            ],
            'dU': [],
            'dI': []
        }
    },
    '2': {
        'U0': ["6.000"],
        'smin': ["0.01"],
        's0': ["10.02"],
        'ds': ["0.005"],
        'b': {
            'U': [
                "5.701", "5.675", "3.704", "2.712", "2.110", "1.694", "1.384", "1.131",
                "0.906", "0.680", "0.412", "0.000"
            ],
            's': [
                "10.03", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1", "0.01"
            ],
            'dU': [],
            'dI': [],
            'dR': [],
            'Rl': ["223.6"]
        },
        'a': {
            'U': [
                "6.000", "5.988", "5.395", "4.800", "4.202", "3.603", "3.003", "2.406",
                "1.805", "1.204", "0.599", "0.000"
            ],
            's': [
                "10.03", "10", "9", "8", "7", "6", "5", "4", "3", "2", "1", "0.01"
            ],
            'dU': [],
            'dI': []
        }
    },
    '3': {
        'U-right': {
            'rough': {
                'U': [
                    "0.578", "0.600", "0.640", "0.681", "0.720", "0.762", "0.800",
                    "0.841", "0.881", "0.920"
                ],
                'I': [
                    "0.001", "0.001", "0.003", "0.009", "0.022", "0.058", "0.140",
                    "0.390", "1.05", "2.06"
                ],
                'dU': [],
                'dI': []
            },
            'fine': {
                'U': [
                    "0.381", "0.397", "0.420", "0.441", "0.463", "0.484", "0.506",
                    "0.522", "0.543", "0.563", "0.583", "0.602", "0.620", "0.645",
                    "0.662", "0.681", "0.704", "0.720", "0.745", "0.761", "0.780",
                    "0.800"
                ],
                'I': [
                    "0.01", "0.01", "0.03", "0.05", "0.09", "0.16", "0.26", "0.37",
                    "0.58", "0.88", "1.33", "1.96", "2.84", "4.72", "6.84", "10.10",
                    "16.67", "23.49", "41.2", "60.6", "93.3", "146.7"
                ],
                'dU': [],
                'dI': []
            }
        },
        'I-right': {
            'rough': {
                'U': [
                    "0.589", "0.600", "0.639", "0.682", "0.720", "0.763", "0.800",
                    "0.843", "0.885", "0.925", "0.959", "1.007"
                ],
                'I': [
                    "0.001", "0.001", "0.003", "0.009", "0.021", "0.052", "0.109",
                    "0.222", "0.393", "0.594", "0.754", "1.103"
                ],
                'dU': [],
                'dI': []
            },
            'fine': {
                'U': [
                    "0.386", "0.398", "0.420", "0.439", "0.458", "0.479", "0.499",
                    "0.519", "0.540", "0.562", "0.584", "0.598", "0.622", "0.639",
                    "0.660", "0.682", "0.704", "0.719", "0.742", "0.760", "0.779",
                    "0.800", "0.820"
                ],
                'I': [
                    "0.01", "0.01", "0.02", "0.04", "0.07", "0.11", "0.18", "0.29",
                    "0.45", "0.69", "1.05", "1.35", "2.09", "2.75", "3.84", "5.23",
                    "6.92", "8.25", "10.51", "12.45", "14.71", "17.21", "19.84"
                ],
                'dU': [],
                'dI': []
            }
        }
    }
}


def last_digit(num, n=1):
    if isinstance(num, np.ndarray):
        return np.vectorize(last_digit)(num, n)
    elif isinstance(num, (int, float, Decimal)):
        if isinstance(num, (float, Decimal)):
            return round(n * 10**-LastSignificant(num), LastSignificant(num))
        else: 
            return n
    else:
        raise TypeError("Input must be an int, float, Decimal or np.ndarray")


def calc_dU(U):
    return np.array(U, dtype=float) * 0.0015 + last_digit(U, 2)

def calc_dI(I):
    return np.array(I, dtype=float) * 0.012 + last_digit(I, 3)

def calc_dR(R):
    return np.array(R, dtype=float) * 0.009 + last_digit(R, 2)


# Convert lists to numpy arrays
for key in data:
    for key2 in data[key]:
        if isinstance(data[key][key2], dict):
            for key3 in data[key][key2]:
                if isinstance(data[key][key2][key3], list):
                    data[key][key2][key3] = np.array(data[key][key2][key3])
                elif isinstance(data[key][key2][key3], dict):
                    for key4 in data[key][key2][key3]:
                        data[key][key2][key3][key4] = np.array(data[key][key2][key3][key4])

for key in data['1']:
    U = np.array([Decimal(U) for U in data['1'][key]['U']])
    U0 = max(U)
    I = np.array([Decimal(I) for I in data['1'][key]['I']])
    I0 = max(I)

    
    dU = calc_dU(U)
    U = np.array(U, dtype=float)
    dU0 = calc_dU(U0)
    U0 = float(U0)
    dI = calc_dI(I)
    I = np.array(I, dtype=float)
    dI0 = calc_dI(I0)
    I0 = float(I0)


    print(U)
    print(dU)

    data['1'][key]['dU'] = np.where(U == 0, 0, U/U0 * np.sqrt((dU / U) ** 2 + (dU0 / U0) ** 2) ) 
    data['1'][key]['dI'] = np.where(I == 0, 0, I/I0 * np.sqrt((dI / I) ** 2 + (dI0 / I0) ** 2) ) 
    data['1'][key]['U'] = U / U0
    data['1'][key]['I'] = I / I0


spline_R = UnivariateSpline(data['1']['R']['U'], data['1']['R']['I'], s=2)
spline_Lamp = UnivariateSpline(data['1']['Lamp']['U'], data['1']['Lamp']['I'], s=0, k=3)
spline_Graphite = UnivariateSpline(data['1']['Graphite']['U'], data['1']['Graphite']['I'], s=2)



plot_data(
    datasets=[
        {
            'xdata': data['1']['R']['U'],
            'ydata': data['1']['R']['I'],
            'x_error': data['1']['R']['dU'],
            'y_error': data['1']['R']['dI'],
            'label': 'R',
            'marker': '.',
            'line': 'None',
            'fit': spline_R(data['1']['R']['U']),
        },
        {
            'xdata': data['1']['Lamp']['U'],
            'ydata': data['1']['Lamp']['I'],
            'x_error': data['1']['Lamp']['dU'],
            'y_error': data['1']['Lamp']['dI'],
            'label': 'Lamp',
            'marker': '.',
            'line': 'None',
            'fit': spline_Lamp(np.linspace(0, 1, 100)),
            'high_res_x': np.linspace(0, 1, 100),
        },
        {
            'xdata': data['1']['Graphite']['U'],
            'ydata': data['1']['Graphite']['I'],
            'x_error': data['1']['Graphite']['dU'],
            'y_error': data['1']['Graphite']['dI'],
            'label': 'Graphite',
            'marker': '.',
            'line': 'None',
            'fit': spline_Graphite(data['1']['Graphite']['U']),
        },
    ],
    x_label='U/Umax',
    y_label='I/Imax',
    title='1',
    filename=f'Plots/ELM_1.pdf',
    color_seed=1,
    plot=False
)


for key in data['2']:
    if isinstance(data['2'][key], dict):

        U = np.array([Decimal(U) for U in data['2'][key]['U']])
        U0 = max(U)
        s = np.array([float(s) for s in data['2'][key]['s']])
        s0 = float(data['2']['s0'][0])
        smin = float(data['2']['smin'][0])

        dU = calc_dU(U)
        U = np.array(U, dtype=float)
        dU0 = calc_dU(U0)
        U0 = float(U0)

        data['2'][key]['dU'] = np.where(U == 0, 0, U/U0 * np.sqrt((dU / U) ** 2 + (dU0 / U0) ** 2) )
        data['2'][key]['U'] = U / U0
        data['2'][key]['s'] = (s - smin) / s0
        data['2']['ds'] = [float(item) for item in data['2']['ds']]
            


def fit_belastet(x, Rl):
    return x / (1 + x*(1-x)*Rl)

plot_data(
    datasets=[
        {
            'xdata': data['2']['b']['s'],
            'ydata': data['2']['b']['U'],
            'x_error': data['2']['ds'],
            'y_error': data['2']['b']['dU'],
            'label': 'belastet',
            'marker': '.',
            'fit': fit_belastet(np.linspace(0, 1, 100), 950/float(data['2']['b']['Rl'][0])),
            'high_res_x': np.linspace(0, 1, 100),
        },
        {
            'xdata': data['2']['a']['s'],
            'ydata': data['2']['a']['U'],
            'x_error': data['2']['ds'],
            'y_error': data['2']['a']['dU'],
            'label': 'unbelastet',
            'marker': '.',
            'fit': np.linspace(0, 1, 100),
            'high_res_x': np.linspace(0, 1, 100),
        },
    ],

    x_label='s/s0',
    y_label='U/U0',
    title='2',
    filename=f'Plots/ELM_2.pdf',
    color_seed=1,
    plot=False
)

for key in data['3']:
    for key2 in data['3'][key]:
        U = np.array([Decimal(U) for U in data['3'][key][key2]['U']])
        I = np.array([Decimal(I) for I in data['3'][key][key2]['I']])

        data['3'][key][key2]['dU'] = calc_dU(U)
        data['3'][key][key2]['dI'] = calc_dI(I)
        data['3'][key][key2]['U'] = np.array(U, dtype=float)
        data['3'][key][key2]['I'] = np.array(I, dtype=float)

plot_data(
    datasets=[
        {
            'xdata': data['3']['U-right']['rough']['U'],
            'ydata': data['3']['U-right']['rough']['I'] * 1000,
            'x_error': data['3']['U-right']['rough']['dU'],
            'y_error': data['3']['U-right']['rough']['dI'] * 1000,
            'label': 'U-richtig grob',
            'marker': '.',
            'line': 'None',
            'color_group': '1',
        },
        {
            'xdata': data['3']['U-right']['rough']['U'],
            'ydata': (data['3']['U-right']['rough']['I'] - data['3']['U-right']['rough']['U']/(11.11e6)) * 1000,
            'x_error': data['3']['U-right']['rough']['dU'],
            'y_error': data['3']['U-right']['rough']['dI'] * 1000,
            'label': 'U-richtig grob korrigiert',
            'marker': '.',
            'line': 'None',
            'color_group': '1',
        }, 
        {
            'xdata': data['3']['U-right']['fine']['U'],
            'ydata': data['3']['U-right']['fine']['I'],
            'x_error': data['3']['U-right']['fine']['dU'],
            'y_error': data['3']['U-right']['fine']['dI'],
            'label': 'U-richtig fein',
            'marker': '.',
            'line': 'None',
            'color_group': '2',
        },
        {
            'xdata': data['3']['U-right']['fine']['U'],
            'ydata': data['3']['U-right']['fine']['I'] - data['3']['U-right']['fine']['U']/(11.11e6) * 1000,
            'x_error': data['3']['U-right']['fine']['dU'],
            'y_error': data['3']['U-right']['fine']['dI'],
            'label': 'U-richtig fein korrigiert',
            'marker': '.',
            'line': 'None',
            'color_group': '2',
        },
        {
            'xdata': data['3']['I-right']['rough']['U'],
            'ydata': data['3']['I-right']['rough']['I']*1000,
            'x_error': data['3']['I-right']['rough']['dU'],
            'y_error': data['3']['I-right']['rough']['dI']*1000,
            'label': 'I-richtig grob',
            'marker': '.',
            'line': 'None',
            'color_group': '3',
        },
            {
            'xdata': data['3']['I-right']['rough']['U'] - data['3']['I-right']['rough']['I']*0.1,
            'ydata': (data['3']['I-right']['rough']['I']) * 1000,
            'x_error': data['3']['I-right']['rough']['dU'],
            'y_error': data['3']['I-right']['rough']['dI'] * 1000,
            'label': 'I-richtig grob korrigiert',
            'marker': '.',
            'line': 'None',
            'color_group': '3',
        },
        {
            'xdata': data['3']['I-right']['fine']['U'],
            'ydata': data['3']['I-right']['fine']['I'],
            'x_error': data['3']['I-right']['fine']['dU'],
            'y_error': data['3']['I-right']['fine']['dI'],
            'label': 'I-richtig fein',
            'marker': '.',
            'line': 'None',
            'color_group': '4',
        },
        {
            'xdata': data['3']['I-right']['fine']['U'] - data['3']['I-right']['fine']['I']*3.9/1000,
            'ydata': data['3']['I-right']['fine']['I'],
            'x_error': data['3']['I-right']['fine']['dU'],
            'y_error': data['3']['I-right']['fine']['dI'],
            'label': 'I-richtig fein korrigiert',
            'marker': '.',
            'line': 'None',
            'color_group': '4',
        },
    ],
    x_label='U/V',
    y_label='I/mA',
    title='3',
    filename=f'Plots/ELM_3.pdf',
    color_seed=37,
    plot=True
)

plot_data(
    datasets=[
        {
            'xdata': data['3']['U-right']['rough']['U'],
            'ydata': np.log(data['3']['U-right']['rough']['I']* 1000),
            'label': 'U-richtig grob',
            'marker': 'x',
            'line': 'None',
        }, 
        {
            'xdata': data['3']['U-right']['rough']['U'],
            'ydata': np.log((data['3']['U-right']['rough']['I'] - data['3']['U-right']['rough']['U']/(11.11e6)) * 1000),
            'label': 'U-richtig grob korrigiert',
            'marker': 'x',
            'line': 'None',
        }, {},{},
        {
            'xdata': data['3']['U-right']['fine']['U'],
            'ydata': np.log(data['3']['U-right']['fine']['I']),
            'label': 'U-richtig fein',
            'marker': '.',
            'line': 'None',
        }, 
        {
            'xdata': data['3']['U-right']['fine']['U'],
            'ydata': np.log(data['3']['U-right']['fine']['I'] - data['3']['U-right']['fine']['U']/(11.11e6)),
            'label': 'U-richtig fein korrigiert',
            'marker': '.',
            'line': 'None',
        }, {},{},
        {
            'xdata': data['3']['I-right']['rough']['U'],
            'ydata': np.log(data['3']['I-right']['rough']['I']*1000),
            'label': 'I-richtig grob',
            'marker': 'x',
            'line': 'None',
        },
        {
            'xdata': data['3']['I-right']['rough']['U'] - data['3']['I-right']['rough']['I']*0.1,
            'ydata': np.log(data['3']['I-right']['rough']['I']*1000),
            'label': 'I-richtig grob korrigiert',
            'marker': 'x',
            'line': 'None',
        }, {},{},
        {
            'xdata': data['3']['I-right']['fine']['U'],
            'ydata': np.log(data['3']['I-right']['fine']['I']),
            'label': 'I-richtig fein',
            'marker': '.',
            'line': 'None',
        },
        {
            'xdata': data['3']['I-right']['fine']['U'] - data['3']['I-right']['fine']['I']*3.9/1000,
            'ydata': np.log(data['3']['I-right']['fine']['I']),
            'label': 'I-richtig fein korrigiert',
            'marker': '.',
            'line': 'None',
        },
    ],
    x_label='U/V',
    y_label='ln(I)/ln(mA)',
    title='3-log',
    filename=f'Plots/ELM_3_log.pdf',
    color_seed=143,
    plot=False
)
