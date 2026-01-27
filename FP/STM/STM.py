import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks


def load_gwyddion_profiles(fname, sep=';'):
    df = pd.read_csv(fname, sep=sep, header=None)

    def is_float(s):
        try:
            float(s)
            return True
        except Exception:
            return False

    start_idx = None
    for i, v in enumerate(df[0]):
        if is_float(str(v)):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError("No numeric data in column 0.")

    header_row = df.iloc[0]
    units_row  = df.iloc[1]
    data       = df.iloc[start_idx:].reset_index(drop=True)

    x_all = pd.to_numeric(data[0], errors='coerce').to_numpy()

    profiles = []
    for col in data.columns[1:]:
        y_raw = pd.to_numeric(data[col], errors='coerce').to_numpy()
        mask  = np.isfinite(x_all) & np.isfinite(y_raw)
        if mask.sum() < 2:
            continue
        profiles.append(dict(
            name=str(header_row[col]),
            unit=str(units_row[col]),
            x=x_all[mask],
            y=y_raw[mask]
        ))
    return profiles


fname = Path("FP/STM/Data/8fb.csv")  
profiles = load_gwyddion_profiles(fname, sep=';')
print(f"Loaded {len(profiles)} profiles")


plt.figure(figsize=(8, 4))

for profile in profiles:
    x = profile['x']
    y = profile['y']
    plt.scatter(x, y, label=f"Profile: {profile['name']}")

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 4))

all_terraces = []

for i in range(6):
    print(i)
    fname = Path(f"FP/STM/Data/8fb{i+1}.csv")
    terraces = load_gwyddion_profiles(fname, sep=';')
    
    terrace = []
    for t in terraces:
        terrace.append(t['y'][0])
    print(terrace)
    
    all_terraces.append(terrace)
    plt.scatter(np.arange(len(terrace)) + 0.5, terrace, label=f"Terrace set {i+1}")

# Calculate average for each index
max_len = max(len(t) for t in all_terraces)
averages = []
averages_sterror = []
for idx in range(max_len):
    values = [t[idx] for t in all_terraces if idx < len(t)]
    averages.append(np.mean(values))
    averages_sterror.append(np.std(values) / np.sqrt(len(values)))

print(f"Averages for each index: {averages} ± {averages_sterror}")
diffs = np.diff(averages) * 1e10
errs = np.sqrt(np.array(averages_sterror[:-1])**2 + np.array(averages_sterror[1:])**2) * 1e10

if diffs.size == 0:
    print("No consecutive averages to compare.")
else:
    for i, (d, e) in enumerate(zip(diffs, errs), start=1):
        print(f"Δavg[{i-1}->{i}]: ({d:.6g} ± {e:.6g}) Å, Which is ({(d/3.35):.6g} ± {(e/3.35):.6g}) multiples of graphite interlayer distance.")
        
edges = np.arange(len(averages) + 1)
plt.stairs(averages, edges, color='black', label='Average (plateaus)')
upper = np.array(averages) + np.array(averages_sterror)
lower = np.array(averages) - np.array(averages_sterror)
plt.fill_between(edges, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]], step='post', alpha=0.2, color='black', label='Std. error')


# Auto y-range: compute from all plotted y-values, add a small padding
y_points = np.array([val for terrace in all_terraces for val in terrace], dtype=float)
y_all = np.concatenate([y_points, upper, lower]) if y_points.size else np.concatenate([upper, lower])
ymin = np.nanmin(y_all)
ymax = np.nanmax(y_all)
if np.isfinite(ymin) and np.isfinite(ymax):
    pad = 0.05 * max(ymax - ymin, 1e-12)
    plt.ylim(ymin - pad, ymax + pad)

plt.xlabel('Index')
plt.ylabel('Y')
plt.legend()
plt.show()
    
    