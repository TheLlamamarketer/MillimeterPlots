import numpy as np
import matplotlib.pyplot as plt

T = {
    'Ar_T_50': [16.32, 16, 16.15, 15.98, 16.19, 16.03, 15.98, 15.97, 16.11, 16.05],
    'Ar_T_100': [32.24, 32.19, 31.93, 32.15, 32.09, 32.24, 31.97, 32.04, 32.07, 32.12],
    'CO2_T_50': [18.69, 18.75, 18.62, 18.67, 18.43, 18.88, 18.64, 18.7, 18.63, 18.81],
    'CO2_T_100': [37.68, 37.46, 37.32, 37.41, 37.36, 37.47, 37.51, 37.38, 37.49, 37.61],
    'N2_T_50': [17.2, 16.25, 17.05, 16.27, 17.12, 16.87, 17.22, 16.92, 17.25, 16.5], # 17.09, 17.14
    'N2_T_100': [34.42, 34.01, 33.47, 34, 33.67, 33.98, 33.63, 33.69, 33.76, 33.63]
}

gas = ['Ar', 'CO2', 'N2']
periods = [50, 100]


m = {'Ar': 4.52e-3, 'CO2': 4.5e-3, 'N2': 4.58e-3}
V = {'Ar': 1142e-6, 'CO2': 1146e-6, 'N2': 1145e-6}
P = 1e5
d = 11.9e-3

dV = 3e-6
dm = 0.01e-3
dd = 0.03e-3
dP = 0.01e5

dT_p = {f'{g}_T_{p}': np.sqrt(sum((T[f'{g}_T_{p}'][i] - np.mean(T[f'{g}_T_{p}'][:10]))**2 for i in range(10)) / (90))  for g in gas for p in periods}
dT = {f'{g}_T_{p}': np.sqrt(sum((T[f'{g}_T_{p}'][i] - np.mean(T[f'{g}_T_{p}'][:10]))**2/p for i in range(10)) / (90))  for g in gas for p in periods}

T_avg = {f'{g}_{p}': sum(T[f'{g}_T_{p}'][:10]) / (len(T[f'{g}_T_{p}'][:10])*p) for g in gas for p in periods}


k = {f'{g}_{p}': (64 * m[g] * V[g]) / (d**4 * P * (T_avg[f'{g}_{p}'])**2) for g in gas for p in periods}

gas = ['Ar', 'CO2', 'N2']
dk = {f'{g}_{p}': k[f'{g}_{p}'] * np.sqrt(
    (dm / m[g])**2 + 
    (dV / V[g])**2 + 
    (4 * dd / d)**2 + 
    (dP / P)**2 + 
    (2 * dT[f'{g}_T_{p}'] / T_avg[f'{g}_{p}'])**2  
    ) for g in gas for p in periods}


f = {'Ar': 3, 'CO2': 7, 'N2': 5}
k_th = {g: (f[g] + 2) / f[g] for g in f}

gas = ['Ar', 'CO2', 'N2']

for g in gas:
    for p in periods:
        key = f'{g}_{p}'
        print(f"$\\kappa_{{{g}_{{{p}}}}} = {k[key]:.6f} \\pm {dk[key]:.6f}$")

k_avg = {g: np.average([k[f'{g}_{p}'] for p in periods]) for g in gas}
dk_avg = {g: np.sqrt(sum([dk[f'{g}_{p}']**2 for p in periods])) / len(periods) for g in gas}

print()
gas = ['Ar', 'CO2', 'N2']
for g in gas:
    print(f"$\\kappa_{{{g}}} = {k_avg[g]:.3f} \\pm {dk_avg[g]:.3f} $")


print()
print("Theoretische Werte")
for gas, value in k_th.items():
    print(f"$\\kappa_{{\\text{{{gas}}}}}^{{\\ast}} = {value:.3f}$")


print()

def print_mini_table(m, V, gas):
    print("\\begin{table}[h!]")
    print("    \\centering")
    print("    \\caption{Gerätedaten über die drei verwendeten Gasoszillatoren (für jede Gasflasche einen Aufbau mit Arbeitsvolumen V, Kolbenmasse m)}")
    print("    \\sisetup{table-format=2.2}")
    print("    \\begin{tabular}{| *{2}{S[table-format=2.2]} | *{2}{S[table-format=2.2]} | *{2}{S[table-format=2.2]} |}")
    print("    \\toprule")
    print("    \\multicolumn{2}{|c|}{Ar} & \\multicolumn{2}{c|}{CO2} & \\multicolumn{2}{c|}{N2} \\\\")
    print("    \\midrule")
    print("    \\midrule")
    header = " & ".join(["{$m$ (g)} & {$V$ (cm$^3$)}" for _ in gas]) 
    print(f"    {header} \\\\")
    print("    \\midrule")

    row_data = f"{m['Ar']*10**3:.2f} & {V['Ar']*10**6:.0f} & {m['CO2']*10**3:.2f} & {V['CO2']*10**6:.0f} & {m['N2']*10**3:.2f} & {V['N2']*10**6:.0f}"
    print("    " + row_data + " \\\\")
    print("    \\midrule")
    print("    \\midrule")

    last = " & ".join(["{$\\Delta m$ (g)} & {$\\Delta V$ (cm$^3$)}" for _ in gas]) 
    print(f"    {last} \\\\")
    print("    \\midrule")

    final = " & ".join([f"{dm*10**3:.2f} & {dV*10**6:.0f}" for _ in gas]) 
    print(f"    {final} \\\\")
    
    print("    \\bottomrule")
    print("    \\end{tabular}")

    print("    \\label{tab:data}")
    print("\\end{table}")

gas = ['Ar', 'CO2', 'N2']
#print_mini_table(m, V, gas)

def print_result_table(k, dk, gas, periods):
    print("\\begin{table}[h!]")
    print("    \\centering")
    print("    \\caption{Berechnete Werte für die Isotropenexponenten $\kappa_{Gas}$ und ihre Fehler $\Delta \kappa_{Gas}$ für die Gase Argon (Ar), Kohlenstoffdioxid (CO2) und Stickstoff (N2) nach Auswertung der Gasoszillator Messwerte für die Dauer T von 50 bzw. 100 Perioden}")
    print("    \\sisetup{table-format=2.3}")
    print("    \\begin{tabular}{|l | *{2}{S[table-format=2.3]} | *{2}{S[table-format=2.3]} | *{2}{S[table-format=2.3]} |}")
    print("    \\toprule")
    header = "$T$ & {$\\kappa_{Ar}$} & {$\\Delta \\kappa_{Ar}$} & {$\\kappa_{CO2}$} & {$\\Delta \\kappa_{CO2}$} & {$\\kappa_{N2}$} & {$\\Delta \\kappa_{N2}$}"
    print(f"    {header} \\\\")
    print("    \\midrule")
    
    for p in periods:
        row_data = " & ".join([f"{k[f'{g}_{p}']:.3f} & {dk[f'{g}_{p}']:.3f}" for g in gas])
        row_str = f"{p} & {row_data}"
        print(f"    {row_str} \\\\")
    
    print("    \\bottomrule")
    print("    \\end{tabular}")
    print("    \\label{tab:results}")
    print("\\end{table}")

print_result_table(k, dk, gas, periods)

def print_table(T):
    num_rows = len(T['Ar_T_50'])  # Assuming all lists have the same length

    print("\\begin{table}[h!]")
    print("    \\centering")
    print("    \\caption{Messdaten Gasoszillator für Argon (Ar) Kohlenstoffdioxid (CO2) und Stickstoff (N2)($T_j$ ist die Dauer von j Perioden) und einem Außenluftdruck von $p_A=(1\pm0,001)$bar}")
    print("    \\sisetup{table-format=2.2}")
    print("    \\begin{tabular}{| *{2}{S[table-format=2.2]} | *{2}{S[table-format=2.2]} | *{2}{S[table-format=2.2]} |}")
    print("    \\toprule")
    print("    \\multicolumn{2}{|c|}{Ar} & \\multicolumn{2}{c|}{CO2} & \\multicolumn{2}{c|}{N2} \\\\")
    print("    \\midrule")
    header = "{$T_{50}$ (s)} & {$T_{100}$ (s)} & {$T_{50}$ (s)} & {$T_{100}$ (s)} & {$T_{50}$ (s)} & {$T_{100}$ (s)}"
    print(f"    {header} \\\\")
    print("    \\midrule")
    
    for row in range(num_rows):
        Ar_T_50 = T['Ar_T_50'][row]
        Ar_T_100 = T['Ar_T_100'][row]
        CO2_T_50 = T['CO2_T_50'][row]
        CO2_T_100 = T['CO2_T_100'][row]
        N2_T_50 = T['N2_T_50'][row]
        N2_T_100 = T['N2_T_100'][row]
        row_data = f"{Ar_T_50:.2f} & {Ar_T_100:.2f} & {CO2_T_50:.2f} & {CO2_T_100:.2f} & {N2_T_50:.2f} & {N2_T_100:.2f}"
        print("    " + row_data + " \\\\")
    print(f"    \\midrule")


    print(f"    \\midrule")
    first = " & ".join([f"{{$\\bar{{T}}_{{{p}}}$ (s)}}" for _ in gas for p in periods]) 
    print(f"    {first} \\\\")
    print(f"    {T_avg['Ar_50']*50:.3f} & {T_avg['Ar_100']*100:.3f} & {T_avg['CO2_50']*50:.3f} & {T_avg['CO2_100']*100:.3f} & {T_avg['N2_50']*50:.2f} & {T_avg['N2_100']*100:.3f} \\\\")
    print(f"    \\midrule")
    middle = " & ".join([f"{{$\\Delta\\bar{{T}}_{{{p}}}$ (s)}}" for _ in gas for p in periods]) 
    print(f"    {middle} \\\\")
    print(f"    {dT_p['Ar_T_50']:.3f} & {dT_p['Ar_T_100']:.3f} & {dT_p['CO2_T_50']:.3f} & {dT_p['CO2_T_100']:.3f} & {dT_p['N2_T_50']:.2f} & {dT_p['N2_T_100']:.3f} \\\\")
    print(f"    \\midrule")


    bottom = " & ".join([f"{{$\\bar{{T}}$ (s)}}" for _ in gas for p in periods]) 
    print(f"    {bottom} \\\\")
    print(f"    {T_avg['Ar_50']:.4f} & {T_avg['Ar_100']:.4f} & {T_avg['CO2_50']:.4f} & {T_avg['CO2_100']:.4f} & {T_avg['N2_50']:.3f} & {T_avg['N2_100']:.4f} \\\\")
    print(f"    \\midrule")

    last = " & ".join([f"{{$\\Delta\\bar{{T}}$ (s)}}" for _ in gas for p in periods]) 
    print(f"    {last} \\\\")
    print(f"    {dT['Ar_T_50']:.4f} & {dT['Ar_T_100']:.4f} & {dT['CO2_T_50']:.4f} & {dT['CO2_T_100']:.4f} & {dT['N2_T_50']:.3f} & {dT['N2_T_100']:.4f} \\\\")
    print("    \\bottomrule")
    print("    \\end{tabular}")
    print("    \\label{tab:A2}")
    print("\\end{table}")

#print_table(T)

