import numpy as np
import matplotlib.pyplot as plt
from SPW_Const import dm_w

data = {
    't': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'T': [0.8, 3, 2.4, 2.5, 2.7, 2.7, 2.8, 2.8, 2.8, 2.8, 2.8]
}

error = [i * 0.001 + 0.2 for i in data['T']]


T_m = 275.825
T_w = 0.8 + 273.15
T_k = 24.5 + 273.15

dT_m = 0.19
dT_w = (T_w - 273.15) * 0.001 + 0.2
dT_k = 0.5


m_w = 0.30481

m_Ä = m_w * (T_m - T_w) / (T_k - T_w)

m = m_w + m_Ä

dm_Ä = m_Ä * np.sqrt(
    (dm_w/m_w)**2 + 
    (dT_w / (T_m - T_w))**2 + 
    (dT_k / (T_m - T_k))**2 + 
    (dT_m * (T_k - T_m) / ((T_k - T_m)*(T_m - T_w)) )**2
)

dm = np.sqrt(dm_Ä**2 + dm_w**2)


print()

print(f"$T_m = ({T_m:.4f} \\pm {dT_m:.4f})$ K")
print(f"$T_w = ({T_w:.4f} \\pm {dT_w:.4f})$ K")
print(f"$T_k = ({T_k:.4f} \\pm {dT_k:.4f})$ K")
print()
print(f"$m_A = ({m_Ä:.8f} \\pm {dm_Ä:.8f})$ kg")
print(f"m = ({m:.8f} \\pm {dm:.8f})$ kg")


def print_table(data, error):
    print("\\begin{table}[h!]")
    print("    \\centering")
    print("    \\caption{ Messdaten Vorversuch: Aufwärmprozess von kaltem Wasser in einem Kalorimeter bei Raumtemperatur (t ist der Zeitpunkt der Messung nach Einfüllen des Wassers, T ist die Temperatur des Wassers, die das Thermometer zum Zeitpunkt t misst). In Abbildung \\ref{fig:A0} grafisch dargestellt.}")
    print("    \\sisetup{table-format=2.2}")
    print("    \\begin{tabular}{| l |*{2}{S[table-format=2.2]} |}")
    print("    \\toprule")
    print("    {$t$ (s)} & {$T$ (°C)} & {$\\Delta T$ (°C)} \\\\")
    print("    \\midrule") 
    for t, T, e in zip(data['t'], data['T'], error):
        print(f"    {t:.0f} & {(T+0):.1f} & {e:.3f} \\\\")
    print("    \\bottomrule")
    print("    \\end{tabular}")
    print("    \\label{tab:A0}")
    print("\\end{table}")

#print_table(data, error)

#plt.xlabel('Time (s)')
#plt.ylabel('Temperature (°C)')
#plt.title('Temperature vs Time')
#plt.errorbar(data['t'], data['T'], color='#51b2e2', linestyle='dotted', marker='x', yerr=error, capsize=5)
#plt.show()