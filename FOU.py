import numpy as np
import matplotlib.pyplot as plt
import cv2
from plotting_minus import plot_data
from lmfit import Model
from help import *
from scipy.stats import f as f_dist


# Load the image in grayscale
file = 'G3'
img = cv2.imread(f'FOU_data/{file}.png', cv2.IMREAD_UNCHANGED)
print(img.shape, img.dtype, img.min(), img.max())

red_chan = img[:,:,2]
cmap_yell = plt.cm.colors.LinearSegmentedColormap.from_list('custom_yellow', ['black', 'yellow'], N=256)


green_chan = img[:,:,1]
arrow_mask = (green_chan > 50) & (red_chan > 50)

green_working = green_chan.astype(float)
green_working[arrow_mask] = np.nan

slices = []
range_slices = (0, 1280)

slices = []
for i in range(range_slices[0], range_slices[1]):
    vertical_slice = green_working[:, i]
    normalized_slice = vertical_slice / 255.0 
    slices.append(normalized_slice)

average_slice = np.nanmean(slices, axis=0)
std_slice = np.nanstd(slices, axis=0)

t_data = np.arange(len(average_slice))

# readd old autocorrelation

def autocorrelation(x):
    results = []
    for k in range(0, len(x)):
        result = np.sum(x[k:] * x[:len(x)-k])/np.sqrt(np.sum(x[k:]**2) * np.sum(x[:len(x)-k]**2))
        results.append(result)
    return np.array(results)

def find_extrema(x):
    maxima = [0]
    minima = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            maxima.append(i)
        if x[i] < x[i - 1] and x[i] < x[i + 1]:
            minima.append(i)
    return maxima, minima

autocor_res = autocorrelation(average_slice)
maxima, minima = find_extrema(autocor_res)

print(f"Distances between maxima: {np.diff(maxima)}")
print(f"Distances between minima: {np.diff(minima)}")


data = {
    'G5':[7.5, 22.5, 37, 50.5, 64],
    'dG5':[1, 2, 2, 3, 3],
    'lambda5': 65.35714286,
    'G4':[5.5, 15.5, 26.5, 36, 47],
    'dG4':[1, 2, 1.5, 2.5, 2.5],
    'lambda4': 92.55,
    'G3':[4, 11.5, 19, 26, 33.5],
    'dG3':[0.75, 1, 1.5, 2, 2.5],
    'lambda3': 130.5,
    'G2':[3, 8, 14, 18.5, 24],# 29, 34],
    'dG2':[0.5, 2, 1, 2, 1.5],# 2, 2],
    'lambda2': 185,
    'G1':[2, 6, 9, 13, 16.5],# 20.5, 24.5, 28],
    'dG1':[0.5, 0.5, 1.5, 1.5, 2],# 1.5, 2, 2],
    'lambda1': 261,
    'dlambda': 0.5
}

for key in data:
    if isinstance(data[key], list):
        data[key] = np.array(data[key])
        data[key] = data[key]/2


plot_data(
    datasets=[
        {
            'xdata': t_data,
            'ydata': average_slice,
            'label': 'Intensity Profile',
            'line': '-',
            'marker': None,
            'confidence': [(average_slice - std_slice , average_slice + std_slice), (average_slice - 2*std_slice, average_slice + 2*std_slice)]
        },
    ],
    x_label='Pixel Row',
    y_label='Intensity',
    title='Intensity Profile with Error Bounds',
    filename=f'Plots/FOU_{file}.pdf',
    width=25,
    height=10,
    plot=False,
)

plot_data(
    datasets=[
        {
            'xdata': t_data,
            'ydata': autocor_res,
            'label': 'Autocorrelation',
            'line': '-',
            'marker': None,
        },
    ],
    x_label='Pixel Row',
    y_label='Autocorrelation',
    title='Autocorrelation of Intensity Profile',
    filename=f'Plots/FOU_{file}_autocorrelation.pdf',
    width=25,
    height=10,
    plot=False,
)


lambdas = [data['lambda1'], data['lambda2'], data['lambda3'], data['lambda4'], data['lambda5']]

datasets = []
max_length = max(len(data['G1']), len(data['G2']), len(data['G3']), len(data['G4']), len(data['G5']))

for i in range(max_length):
    xdata = []
    ydata = []
    yerr = []
    for G_key, lambda_value in zip(['G1', 'G2', 'G3', 'G4', 'G5'], lambdas):
        G_values = data[G_key]
        if len(G_values) > i:
            ydata.append(G_values[i])
            xdata.append(lambda_value)
            yerr.append(data[f'd{G_key}'][i])

    datasets.append({
        'xdata': xdata,
        'ydata': ydata,
        'y_error': yerr,
        'line': 'None',
        'marker': '.'
    })

plot_data(
    datasets=datasets,
    x_label='grating spacing',
    y_label='Grating frequency',
    title='Grating frequency vs grating spacing',
    filename='Plots/FOU_data1.pdf',
    width=25,
    height=10,
    plot=False,
)
# Calculate xy and dxy
xy = [np.array(dataset['ydata']) * np.array(dataset['xdata']) for dataset in datasets]
dxy = [xy[i] * np.sqrt((np.array(datasets[i]['y_error']) / np.array(datasets[i]['ydata']))**2 + (data['dlambda']/ np.array(datasets[i]['xdata']))**2) for i in range(len(datasets))]

# Fit the data for each dataset
fits = []
for i in range(len(datasets)):
    result = linear_fit(np.array(datasets[i]['xdata']), xy[i], dxy[i], model="linear")
    fits.append(result)

# Print fit reports
for i, result in enumerate(fits):
    params = extract_params(result)
    b, db = params['b']
    b, db, _ = round_val(b, db)
    s2 = result.redchi  
    print(f"Slope (b_{i+1}) = {b} ± {db}. R^2 = {calc_R2(result)}. s^2 = {s2}")


# Prepare datasets for plotting
plot_datasets = []
for i in range(len(datasets)):
    high_res_x = np.linspace(np.min(datasets[i]['xdata']), np.max(datasets[i]['xdata']), 300)
    fit = fits[i].eval(x=high_res_x)
    confidence = calc_CI(fits[i], high_res_x)
    plot_datasets.append({
        'xdata': datasets[i]['xdata'],
        'ydata': xy[i],
        'y_error': dxy[i],
        'label': f'Dataset {i+1}',
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
    y_label='Product of Grating frequency and spacing (xy)',
    title='Antiproportionality of Grating frequency and spacing',
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

    lx = np.log(xdata)
    ly = np.log(ydata)
    ldy_up = np.log(np.array(ydata) + np.array(yerr)) - np.log(ydata)
    ldy_low = np.log(ydata) - np.log(np.array(ydata) - np.array(yerr))
    ldy = np.average([ldy_up, ldy_low], axis=0)
    
    result = linear_fit(lx, ly, ldy, model="linear")
    high_res_x = np.linspace(lx.min(), lx.max(), 300)
    fit = result.eval(x=high_res_x)
    confidence = calc_CI(result, high_res_x)
    param = extract_params(result)
    params.append(param)

    datasets2.append({
        'xdata': np.log(xdata),
        'ydata': np.log(ydata),
        'y_error': (ldy_low, ldy_up),
        'fit': fit,
        'line': 'None',
        'marker': '.',
        'confidence': confidence,
        'high_res_x': high_res_x
    })




# Calculate the averages of x, y, and yerrors from before and add them to the datasets
log_xdata_avg = np.log(lambdas)
ydata_log_avg = np.mean([dataset['ydata'] for dataset in datasets2 if dataset['ydata'] is not None], axis=0)
yerr_log_lower_avg = np.mean([dataset['y_error'][0] for dataset in datasets2 if dataset['y_error'] is not None], axis=0)
yerr_log_upper_avg = np.mean([dataset['y_error'][1] for dataset in datasets2 if dataset['y_error'] is not None], axis=0)
yerr_log_avg = (yerr_log_lower_avg + yerr_log_upper_avg) / 2
result_avg = linear_fit(log_xdata_avg, ydata_log_avg, yerr_log_avg, model="linear")
high_res_x_avg = np.linspace(log_xdata_avg.min(), log_xdata_avg.max(), 300)
fit_avg = result_avg.eval(x=high_res_x_avg)
confidence_avg = calc_CI(result_avg, high_res_x_avg)
params_avg = extract_params(result_avg)


datasets2.append({
    'xdata': log_xdata_avg,
    'ydata': ydata_log_avg,
    'y_error': (yerr_log_lower_avg, yerr_log_upper_avg),
    'fit': fit_avg,
    'line': 'None',
    'marker': 'x',
    'label': 'Average after log',
    'confidence': confidence_avg,
    'high_res_x': high_res_x_avg
})

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
    'fit': log_fit_avg,
    'line': 'None',
    'marker': 'x',
    'label': 'Average before log',
    'confidence': log_confidence_avg,
    'high_res_x': log_high_res_x_avg
})


b_values = []
db_values = []
for param in params:
    b, db, _ = round_val(param['b'][0], param['b'][1])
    if db >= 10e-5:    
        b_values.append(param['b'][0])
        db_values.append(param['b'][1])
        print(f"Steigung b = {b} ± {db}")

b_avg = np.average(b_values, weights=np.reciprocal(np.array(db_values) ** 2))
db_avg = 1 / np.sqrt(np.sum(np.reciprocal(np.array(db_values) ** 2)))
b_avg, db_avg, _ = round_val(b_avg, db_avg, intermed=False)

print(f"Weighted average of b values: {b_avg} ± {db_avg}")
b, db = params_avg['b']
b, db, _ = round_val(b, db)
print(f"Average of b values after log: {b} ± {db}")

b, db = log_params_avg['b']
b, db, _ = round_val(b, db)
print(f"Average of b values before log: {b} ± {db}")

plot_data(
    datasets=datasets2,
    x_label='Log of grating spacing',
    y_label='Log of Grating frequency',
    title='Log-Log Plot of Grating frequency vs grating spacing',
    filename='Plots/FOU_data2.pdf',
    width=25,
    height=10,
    plot=True,
)



plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom_green', ['black', 'green', 'lime'], N=256)
plt.imshow(red_chan, cmap=cmap_yell)
plt.imshow(green_working, cmap=cmap)
plt.axis('off')
plt.gca().set_position([0, 0, 1, 1])  
plt.savefig(f'FOU_data2/{file}.png', bbox_inches='tight', pad_inches=0, format='png', dpi=100) 
#plt.show()
