import numpy as np
from plotting_minus import plot_data
from help import *

#data = {
#    'G5':[7.5, 22.5, 37, 50.5, 64],
#    'dG5':[1, 2, 2, 3, 3],
#    'lambda5': 65.35714286,
#    'G4':[5.5, 15.5, 26.5, 36, 47],
#    'dG4':[1, 2, 1.5, 2.5, 2.5],
#    'lambda4': 92.55,
#    'G3':[4, 11.5, 19, 26, 33.5],
#    'dG3':[0.75, 1, 1.5, 2, 2.5],
#    'lambda3': 130.5,
#    'G2':[3, 8, 14, 18.5, 24],# 29, 34],
#    'dG2':[0.5, 2, 1, 2, 1.5],# 2, 2],
#    'lambda2': 185,
#    'G1':[2, 6, 9, 13, 16.5],# 20.5, 24.5, 28],
#    'dG1':[0.5, 0.5, 1.5, 1.5, 2],# 1.5, 2, 2],
#    'lambda1': 261,
#    'dlambda': 0.5
#}
#
#for key in data:
#    if isinstance(data[key], list):
#        data[key] = np.array(data[key])
#        data[key] = data[key]/2

data = {
    'G5': [3.75, 11.25, 18.5, 25.25, 32],
    'dG5': [0.5, 1, 1, 1.5, 1.5],
    'lambda5': 65.35714286,
    'G4': [2.75, 7.75, 13.25, 18, 23.5],
    'dG4': [0.5, 1, 0.75, 1.25, 1.25],
    'lambda4': 92.55,
    'G3': [2, 5.75, 9.5, 13, 16.75],
    'dG3': [0.375, 0.5, 0.75, 1, 1.25],
    'lambda3': 130.5,
    'G2': [1.5, 4, 7, 9.25, 12],
    'dG2': [0.25, 1, 0.5, 1, 0.75],
    'lambda2': 185,
    'G1': [1, 3, 4.5, 6.5, 8.25],
    'dG1': [0.25, 0.25, 0.75, 0.75, 1],
    'lambda1': 261,
    'dlambda': 0.5
}



# Convert 'lambdas' and 'dlambda' to NumPy arrays
lambdas = np.array([data['lambda1'], data['lambda2'], data['lambda3'], data['lambda4'], data['lambda5']])
dlambda = np.full_like(lambdas, data['dlambda'])

datasets = []
max_length = max([len(data[f'G{i}']) for i in range(1, 6)])

for i in range(max_length):
    xdata_list = []
    ydata_list = []
    yerr_list = []
    for idx, G_key in enumerate(['G1', 'G2', 'G3', 'G4', 'G5']):
        G_values = data[G_key]
        if len(G_values) > i:
            ydata_list.append(G_values[i])
            xdata_list.append(lambdas[idx])
            yerr_list.append(data[f'd{G_key}'][i])

    xdata = np.array(xdata_list)
    ydata = np.array(ydata_list)
    yerr = np.array(yerr_list)

    datasets.append({
        'xdata': xdata,
        'ydata': ydata,
        'y_error': yerr,
        'x_error': dlambda,
        'label': f'Beugungsordnung {2*i+1}',
        'line': 'None',
        'marker': '.'
    })

plot_data(
    datasets=datasets,
    x_label='Gitterperiode \\Lambda/pixel',
    y_label='Gitterfrequenz k/mm',
    title='Gitterfrequenz in Abhängigkeit der Gitterperiode',
    filename='Plots/FOU_data1.pdf',
    width=25,
    height=25,
    plot=False,
)

# Ensure 'xy' and 'dxy' are NumPy arrays
xy = [datasets[i]['xdata'] * datasets[i]['ydata']/(2*i+1) for i in range(len(datasets))]
dxy = [xy[i] * np.sqrt(
    (datasets[i]['y_error'] / datasets[i]['ydata'])**2 +
    (dlambda / datasets[i]['xdata'])**2) for i in range(len(datasets))]

# Fit the data for each dataset
fits = []
plot_datasets = []
for i in range(len(datasets)):
    result = linear_fit(np.arange(len(datasets[i]['xdata'])), xy[i], dxy[i], model="linear")

    params = extract_params(result)
    b, db = params['b']
    b, db, _ = round_val(b, db)
    s2 = result.redchi  
    print(f"Slope (b_{i+1}) = {b} ± {db}. R^2 = {calc_R2(result)}. s^2 = {s2}")

    high_res_x = np.linspace(0, len(datasets[i]['xdata']) - 1, 300)
    fit = result.eval(x=high_res_x)
    confidence = calc_CI(result, high_res_x, sigmas=[1])
    plot_datasets.append({
        'xdata': np.arange(len(datasets[i]['xdata'])),
        'ydata': xy[i],
        'y_error': dxy[i],
        'label': f'Beugungsordnung {2*i+1}',
        'line': 'None',
        'marker': 'o',
        'fit': fit,
        'confidence': confidence,
        'high_res_x': high_res_x
    })

# Plot the data and the fits using plot_data
plot_data(
    datasets=plot_datasets,
    x_label='index',
    y_label='Gitterfrequenz * Gitterperiode / (2n+1)',
    title='Gitterfrequenz * Gitterperiode / (2n+1) in Abhängigkeit der Beugungsordnung',
    filename='Plots/FOU_xy_plot.pdf',
    width=25,
    height=25,
    plot=True,
)


datasets2 = []
params = []
max_length = max(len(data['G1']), len(data['G2']), len(data['G3']), len(data['G4']), len(data['G5']))

for i in range(max_length):
    xdata = datasets[i]['xdata']
    ydata = datasets[i]['ydata']
    yerr = datasets[i]['y_error']
    xerr = datasets[i]['x_error']

    lx = np.log(xdata)
    ly = np.log(ydata)
    ldy_up = np.log(ydata + yerr) - ly
    ldy_low = ly - np.log(ydata - yerr)
    ldx_up = np.log(xdata + xerr) - lx
    ldx_low = lx - np.log(xdata - xerr)
    ldy = yerr / ydata  

    result = linear_fit(lx, ly, ldy, model="linear")
    high_res_x = np.linspace(lx.min(), lx.max(), 300)
    fit = result.eval(x=high_res_x)
    confidence = calc_CI(result, high_res_x)
    param = extract_params(result)
    params.append(param)

    datasets2.append({
        'xdata': lx,
        'ydata': ly,
        'y_error': (ldy_low, ldy_up),
        'x_error': (ldx_low, ldx_up),
        'label': f'Beugungsordnung {2*i+1}',
        'fit': fit,
        'fit_label': False,
        'line': 'None',
        'marker': '.',
        'confidence': confidence,
        'confidence_label': False,
        'high_res_x': high_res_x
    })


log_xdata_avg = np.log(lambdas)
log_xerr_avg_up = np.log(lambdas + dlambda) - log_xdata_avg
log_xerr_avg_low = log_xdata_avg - np.log(lambdas - dlambda)
log_xerr_avg = np.average([log_xerr_avg_up, log_xerr_avg_low], axis=0)

ydata_avg = np.mean([dataset['ydata'] for dataset in datasets if dataset['ydata'] is not None], axis=0)
all_y_errors = np.array([np.array(dataset['y_error']) for dataset in datasets if dataset['y_error'] is not None])
yerr_avg = np.sqrt(np.sum(all_y_errors**2, axis=0)) / all_y_errors.shape[0]
log_ydata_avg = np.log(ydata_avg)
log_yerr_avg_up = np.log(ydata_avg + yerr_avg) - log_ydata_avg
log_yerr_avg_low = log_ydata_avg - np.log(ydata_avg - yerr_avg)
log_yerr_avg = np.average([log_yerr_avg_up, log_yerr_avg_low], axis=0)

log_result_avg = linear_fit(log_xdata_avg, log_ydata_avg, log_yerr_avg, model="linear")
log_high_res_x_avg = np.linspace(log_xdata_avg.min(), log_xdata_avg.max(), 300)
log_fit_avg = log_result_avg.eval(x=log_high_res_x_avg)
log_confidence_avg = calc_CI(log_result_avg, log_high_res_x_avg)
log_params_avg = extract_params(log_result_avg)

datasets2.append({
    'xdata': log_xdata_avg,
    'ydata': log_ydata_avg,
    'y_error': (log_yerr_avg_low, log_yerr_avg_up),
    'x_error': (log_xerr_avg_low, log_xerr_avg_up),
    'fit': log_fit_avg,
    'fit_label': False,
    'line': 'None',
    'marker': 'x',
    'label': 'Mittelwert vor Logarithmierung',
    'confidence': log_confidence_avg,
    'confidence_label': False,
    'high_res_x': log_high_res_x_avg
})


b_values = []
db_values = []
for param in params:
    b, db, _ = round_val(param['b'][0], param['b'][1])
    a, da, _ = round_val(param['a'][0], param['a'][1])
    if db >= 10e-5:    
        b_values.append(param['b'][0])
        db_values.append(param['b'][1])
        print(f"Steigung b = {b} ± {db}. Achsenabschnitt a = {a} ± {da}")

b_avg = np.average(b_values, weights=np.reciprocal(np.array(db_values) ** 2))
db_avg = 1 / np.sqrt(np.sum(np.reciprocal(np.array(db_values) ** 2)))
b_avg, db_avg, _ = round_val(b_avg, db_avg, intermed=False)

print(f"Geewichteter Mittelwert der Steigung b = {b_avg} ± {db_avg}")

b, db = log_params_avg['b']
b, db, _ = round_val(b, db, intermed=False)
print(f"Steigung mit Mittelung vor Logarithmierung b = {b} ± {db}. R^2 = {calc_R2(log_result_avg)}. s^2 = {log_result_avg.redchi}")

plot_data(
    datasets=datasets2,
    x_label='ln(Gitterperiode)/ln(pixel)',
    y_label='ln(Gitterfrequenz)/ln(mm)',
    title='ln(Gitterfrequenz) in Abhängigkeit von ln(Gitterperiode)',
    filename='Plots/FOU_data2.pdf',
    width=20,
    height=28,
    color_seed=123,
    plot=True,
)
