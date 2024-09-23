import numpy as np
from plotting import plot_data
from find_interval import find_interval

data = {
    'a': [3048, 3436, 1786, 1377, 3301, 80.2, 181],
    'b': [62.8, 63.1, 46.3, 42.1, 60.7, 28.2, 8.33],
    'E': [1.173, 1.333, 0.662, 0.511, 1.275, 0.027, 0.056],
}

#find the slope through linear regression
slope = np.sum((np.array(data['a']) - np.mean(data['a'])) * (np.array(data['E']) - np.mean(data['E']))) / np.sum((np.array(data['E']) - np.mean(data['E']))**2)

print(f"Slope: {slope}")

#find_interval(data['E'], data['a'], name='data', max_increment_height=500, max_increment_width=1)

#plot_data(ymin=0, ymax=3640, ystep=130, xmin=0, xmax=1.4, xstep=0.14, x_label='E/MeV', y_label='N/#', datasets=[{'xdata':data['E'], 'ydata': data['a'] , 'y_error': data['b']}], filename='C:/Users/alexa/OneDrive/Desktop/newfolder/GAM_1.pdf')


data_pb = {
    'd':[3, 6, 9, 15, 18, 12, 15],
    'A':[12296, 9133, 6968, 3502, 2471, 3502, 2471],
}

data_fe= {
    'd':[9.9, 19.9, 30, 35],
    'A':[32365, 13409, 6139, 4310],
}

grey_zones = [
    {'x_val': 12, 'y_val': np.log(3502)},
    {'x_val': 15, 'y_val': np.log(2471)},
]


data_pb = {key: [np.log(x) for x in value] if key.startswith('A') else value for key, value in data_pb.items()}
data_fe = {key: [np.log(x) for x in value] if key.startswith('A') else value for key, value in data_fe.items()}

slope_pb = np.sum((np.array(data_pb['A'][:-2]) - np.mean(data_pb['A'][:-2])) * (np.array(data_pb['d'][:-2]) - np.mean(data_pb['d'][:-2]))) / np.sum((np.array(data_pb['d'][:-2]) - np.mean(data_pb['d'][:-2]))**2)
slope_fe = np.sum((np.array(data_fe['A']) - np.mean(data_fe['A'])) * (np.array(data_fe['d']) - np.mean(data_fe['d']))) / np.sum((np.array(data_fe['d']) - np.mean(data_fe['d']))**2)

print(f"Slope Pb: {slope_pb}")
print(f"Slope Fe: {slope_fe}")

#find_interval(data_pb['d'], data_pb['A'], name='data_pb', max_increment_height=1, max_increment_width=5)
#find_interval(data_fe['d'], data_fe['A'], name='data_fe', max_increment_height=1, max_increment_width=5)

#find_interval([3, 35], [7.75, 10.5], name='data_combi', max_increment_height=2, max_increment_width=5)

#plot_data(ymin=7.75, ymax=9.5, xmin=2.4, xmax=18.4, xstep=1.6, ystep=0.25, xtickoffset=0, grey_zones=grey_zones, datasets=[{'xdata': data_pb['d'], 'ydata': data_pb['A'], 'x_error': [0.1]*len(data_pb['d']), 'color': '#003366'}], filename='C:/Users/alexa/OneDrive/Desktop/newfolder/GAM_2.pdf')
#plot_data(ymin=8, ymax=10.8, xmin=7, xmax=37, xstep=3, ystep=0.1, datasets=[{'xdata': data_fe['d'], 'ydata': data_fe['A'], 'x_error': [0.1]*len(data_fe['d']), 'color': '#006400'}], filename='C:/Users/alexa/OneDrive/Desktop/newfolder/GAM_3.pdf')

#plot_data(ymin=7.7, ymax=10.5, ystep=0.1, xmax=35, xmin=3, xstep=2, grey_zones=grey_zones, x_label='d/mm', y_label='I/ #/s', filename='C:/Users/alexa/OneDrive/Desktop/newfolder/GAM_4.pdf', datasets=[{'xdata': data_pb['d'], 'ydata': data_pb['A'], 'x_error': [0.1]*len(data_pb['d']), 'color': '#003366', 'label':'Blei'}, {'xdata': data_fe['d'], 'ydata': data_fe['A'], 'x_error': [0.1]*len(data_fe['d']), 'color': '#006400', 'label':'Eisen'}] )






import pandas as pd
file_path = 'Current\cs2.csv'
from plotting_plus import plot_data_plus

column_names = ['Kanalnummer', 'Impulse', 'Fit(Impulse)']
df = pd.read_csv(file_path, sep='\t', names=column_names)

data = {
    'Kanalnummer': df['Kanalnummer'].tolist(),  
    'Impulse': df['Impulse'].tolist(),     
}
find_interval(data['Kanalnummer'], data['Impulse'], name='data', max_increment_height=100, max_increment_width=100, width=28, height=20)
#plot_data_plus(ymin=0, ymax=1000, ystep=50, xmin=0, xmax=4000, xstep=500, width=28, height=20, x_label='Kanalnummer', y_label='Impulse',marker='none', datasets=[{'xdata':data['Kanalnummer'], 'ydata': data['Impulse'], 'label':'Daten'}], filename='C:/Users/alexa/OneDrive/Desktop/newfolder/GAM_5.pdf')

# Extract the relevant columns into lists
data = {
    'Kanalnummer': df['Kanalnummer'][1600:2000].tolist(),  
    'Impulse': df['Impulse'][1600:2000].tolist(),
    'Fit(Impulse)': df['Fit(Impulse)'][1600:2000].tolist(),     
}

#plot_data_plus(ymin=0, ymax=980, ystep=70, xmin=1600, xmax=2000, xstep=50, x_label='Kanalnummer', y_label='Impulse', datasets=[{'xdata':data['Kanalnummer'], 'ydata': data['Impulse'], 'label':'Daten'}, {'xdata':data['Kanalnummer'], 'ydata':data['Fit(Impulse)'], 'label':'Fit'}], filename='C:/Users/alexa/OneDrive/Desktop/newfolder/GAM_6.pdf')


data = {
    'Kanalnummer': df['Kanalnummer'][1100:1700].tolist(),  
    'Impulse': df['Impulse'][1100:1700].tolist(),     
}
#find_interval(data['Kanalnummer'], data['Impulse'], name='data', max_increment_height=100, max_increment_width=100)
plot_data_plus(ymin=0, ymax=224, ystep=20, xmin=1100, xmax=1700, xstep=50, x_label='Kanalnummer', y_label='Impulse', datasets=[{'xdata':data['Kanalnummer'], 'ydata': data['Impulse'], 'label':'Daten'}], filename='C:/Users/alexa/OneDrive/Desktop/newfolder/GAM_7.pdf')
