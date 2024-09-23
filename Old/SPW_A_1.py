import numpy as np
import matplotlib.pyplot as plt
from SPW_A_0 import m_Ä, dm_Ä
from SPW_Const import dm_w



data = {
    't': list(range(21)),
    'T': [19.9, 20.6, 21.2, 21.9, 22.6, 23.3, 24, 24.6, 25.3, 25.8, 25.8, 25.9, 26.3, 26.7, 27.2, 27.7, 28.2, 28.7, 29.3, 29.8, 30.4],
    'U': [6.19, 6.21, 6.24, 6.23, 6.21, 6.24, 6.23, 6.21, 6.2, 6.22, 6.23, 6.21, 6.21, 6.2, 6.2, 6.2, 6.21, 6.21, 6.21, 6.22, 6.19],
    'I': [2.3, 2.31, 2.31, 2.31, 2.31, 2.32, 2.31, 2.3, 2.3, 2.31, 2.31, 2.3, 2.3, 2.29, 2.29, 2.28, 2.3, 2.3, 2.3, 2.3, 2.27]
}

error_T = [i * 0.001 + 0.2 for i in data['T']]
error_U = [i * 0.008 + 0.05 for i in data['U']]
error_I = [i * 0.02 + 0.05 for i in data['I']]


def get_decimal_places(value, significant_figures=4):
    if value == 0:
        return 0
    decimal_places = -int(np.floor(np.log10(abs(value)))) + (significant_figures - 1)
    return max(0, decimal_places)

def format_value(value, error, significant_figures=2):
    decimal_places = get_decimal_places(error, significant_figures)
    format_string = f"{{:.{decimal_places}f}}"
    return format_string.format(value)

def print_table(data, error_T, error_U, error_I):
    print("\\begin{table}[h!]")
    print("    \\centering")
    print("    \\caption{Messdaten Aufgabe 1: Erwärmung von Wasser mit Heizelement (t ist der Zeitpunkt nach Beginn der Messung, T ist die Temperatur des Wassers, die das Thermometer zum Zeitpunkt t misst, U ist die Spannung, die zum Zeitpunkt t am Netzgerät anliegt, I ist die Stromstärke, die zum Zeitpunkt t durch das Heizelement fließt). In Abbildungen \\ref{fig:A1} und \\ref{fig:A1-2} grafisch dargestellt. (Messfehler berechnet in \\ref{fehler_1})}")
    print("    \\sisetup{table-format=2.3}")
    print("    \\begin{tabular}{| l | *{2}{S[table-format=2.3]} | *{2}{S[table-format=2.3]} | *{2}{S[table-format=2.3]} |}")
    print("    \\toprule")
    print("    {$t$ (min)} & {$T$ (°C)} & {$\\Delta T$ (°C)} & {$U$ (V)} & {$\\Delta U$ (V)} & {$I$ (A)} & {$\\Delta I$ (A)} \\\\")
    print("    \\midrule")
    for t, T, eT, U, eU, I, eI in zip(data['t'], data['T'], error_T, data['U'], error_U, data['I'], error_I):
        print(f"    {t:.0f} & {T:.1f} & {eT:.3f} & {U:.2f} & {eU:.2f} & {I:.2f} & {eI:.3f} \\\\")
    print("    \\bottomrule")
    print("    \\end{tabular}")
    print("    \\label{tab:A1}")
    print("\\end{table}")

#print_table(data, error_T, error_U, error_I)

m_w = 0.29517


U = np.average(data['U'])
I = np.average(data['I'])

dU =  np.sqrt(sum(np.square(error_U)))/len(error_U)
dU = U * 0.008 + 0.05
dI =  np.sqrt(sum(np.square(error_I)))/len(error_I)
dI = I * 0.02 + 0.05

m = m_w + m_Ä

dm = np.sqrt( (dm_Ä)**2 + (dm_w)**2)


T_A = 8
T_G = 8.88
t_A = 2800/3
t_G = 3040/3

def calculate_c_w(T_A, T_G, t_A, t_G):
    a = t_A / T_A
    a_G = t_G / T_G
    da = a - a_G
    c_w = (U * I * a) / m

    dc_w = c_w * np.sqrt(
        (dU / U)**2 +
        (dI / I)**2 +
        (da / a)**2 +
        (dm / m)**2
    )

    print()
    print(f"$a = ({a:.4f} \\pm {da:.4f})$ s/K")
    print(f"$m = ({m:.8f} \\pm {dm:.8f})$ kg")
    print(f"$U = ({U:.6f} \\pm {dU:.6f})$ V, $I = ({I:.6f} \\pm {dI:.6f})$ A")
    print(f"$c_W = ({c_w / 1000:.6f} \\pm {dc_w / 1000:.6f})$ kJ/(kg·K)")
    return c_w, dc_w

#calculate_c_w(T_A = 8, T_G = 8.88, t_A = 2800/3, t_G=3040/3)

c_w, dc_w, = calculate_c_w(T_A = 4.64, T_G = 4.64, t_A = 1232/3, t_G=1400/3)


a = 4181.8*m/(U*I)
#print(a, -a*20)






plt.ylabel('Time (s)')
plt.xlabel('Temperature (°C)')
plt.title('Temperature vs Time')
plt.errorbar(data['T'], data['t'], color='#51b2e2', linestyle='dotted', marker='x', yerr=error_T, capsize=5)
plt.show()




