from calendar import c
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import BSpline

from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).parent.parent.parent))

from Functions.plotting import *
from Functions.tables import *
from Functions.help import *
from scipy.interpolate import UnivariateSpline


source_dir = Path('FP/QFM/Data')

massrs, signals = [], []
names, sources, pressure = [], [], []


for f in source_dir.glob('*.txt'):
    p = f.stem.find('_p1')
    redo = f.stem.find('_redo') or f.stem.find('_REDO')
    name = f.stem[:p] + f.stem[p+5:]
    if name.find('redo') != -1 or name.find('REDO') != -1:
        name = name.replace('_redo', '').replace('_REDO', '')   
    names.append(name)
    sources.append(f)
    pressure.append(float(f.stem[p+2:p+5])/10)
print(pressure)

mr_U, s_U, name_U, pressure_U = [], [], [], []

for i, (name, source) in enumerate(zip(names, sources)):
    s, mr = pd.read_csv(source_dir / source.name, sep='\t', header=None, decimal=',').values.T
    mr *= 10  # to m/z
    sort_idx = np.argsort(mr)
    mr, s = mr[sort_idx], s[sort_idx]
    s /= pressure[i]
    
    if name.find('Air_U') != -1:
        s_U.append(s)
        mr_U.append(mr)
        name_U.append(name)
    else:
        continue

fig = plt.figure(figsize=(17, 10))
plt.title('Changing U_2 Voltage for Air at different Pressures')

sorting = np.argsort([int(name[5:]) for name in name_U])
mr_U = [mr_U[i] for i in sorting]
s_U = [s_U[i] for i in sorting]
name_U = [name_U[i] for i in sorting]

for mr, s, name in zip(mr_U, s_U, name_U):
    plt.plot(mr, s, label=f'U={name[5:]} eV')
plt.grid()
plt.legend()
plt.show()



for i, (name, source) in enumerate(zip(names, sources)):
    s, mr = pd.read_csv(source_dir / source.name, sep='\t', header=None, decimal=',').values.T
    mr *= 10  # to m/z
    sort_idx = np.argsort(mr)
    mr, s = mr[sort_idx], s[sort_idx]
    s += abs(min(s))
    s = s/np.max(s)
    
    if name.find('Aceton') != -1:
        mask = (mr < 60) 
    elif name.find('Ethanol') != -1:
        mask = (mr < 50)
    elif name.find('Air') != -1:
        mask = (mr < 100)
    elif name.find('Ar') != -1:
        mask = (mr < 50)
    else:
        mask = (mr < 100)
    
    mr, s = mr[mask], s[mask]
    
    
    s_filtered = savgol_filter(s, 31, 3)
    
    peaks, _ = find_peaks(s_filtered, prominence=0.03, distance=20)
    
    fig = plt.figure(figsize=(17, 10))
    plt.title(name)
    plt.plot(mr, s, label=name)
    plt.plot(mr[peaks], s_filtered[peaks], "o", color='tab:green')
    plt.plot(mr, s_filtered, '-', alpha=0.5, color='tab:red')
    plt.grid()
    
    if (i+1) % 1 == 0:
        plt.show()

    massrs.append(mr)
    signals.append(s)





