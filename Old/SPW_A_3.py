from calendar import c
from re import T
from matplotlib.pylab import f
import numpy as np
import matplotlib.pyplot as plt
from SPW_A_0 import m_Ä, dm_Ä
from SPW_Const import dm_w
from SPW_A_1 import c_w, dc_w

data = {
    't': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210],
    'T': [31, 24.1, 21.5, 19.9, 19.2, 18.9, 18.8, 18.9, 18.9, 18.9, 18.9, 18.9, 18.9, 18.9, 19, 19, 19, 19, 19, 19, 19, 19]
}

error = [i * 0.001 + 0.2 for i in data['T']]

def get_decimal_places(value, significant_figures=2):
    if value == 0:
        return 0
    decimal_places = -int(np.floor(np.log10(abs(value)))) + (significant_figures - 1)
    return max(0, decimal_places)

def format_value(value, error, significant_figures=4):
    decimal_places = get_decimal_places(error, significant_figures)
    format_string = f"{{:.{decimal_places}f}}"
    return format_string.format(value)

def print_table(data, error, max_columns=4):
    num_rows = len(data['t'])
    rows_per_column = (num_rows + max_columns - 1) // max_columns  # Calculate the number of rows per column
    split_data = [data['t'][i:i + rows_per_column] for i in range(0, num_rows, rows_per_column)]
    split_temp = [data['T'][i:i + rows_per_column] for i in range(0, num_rows, rows_per_column)]
    split_error = [error[i:i + rows_per_column] for i in range(0, num_rows, rows_per_column)]

    print("\\begin{table}[h!]")
    print("    \\centering")
    print("    \\caption{Messdaten Aufgabe 3: Schmelzen von Eis in warmem Wasser (t ist der Zeitpunkt nach Einschütten des Eises, T ist die Temperatur des Wassers, die das Thermometer zum Zeitpunkt t misst). In Abbildung \\ref{fig:A3} grafisch dargestellt. (Messfehler berechnet in \\ref{fehler_3})}")
    print("    \\sisetup{table-format=2.3}")
    col_format = "| l | " + " || l | ".join(["*{2}{S[table-format=2.3]}"] * max_columns) + " |"
    print(f"    \\begin{{tabular}}{{{col_format}}}")
    print("    \\toprule")
    header = " & ".join(["{$t$ (s)} & {$T$ (°C)} & {$\\Delta T$ (°C)}"] * max_columns)
    print(f"    {header} \\\\")
    print("    \\midrule")
    
    for row in range(rows_per_column):
        row_data = []
        for col in range(max_columns):
            if row < len(split_data[col]):
                t = split_data[col][row]
                T = split_temp[col][row]
                eT = split_error[col][row]
                formatted_T = format_value(T, eT)
                formatted_eT = format_value(eT, eT)
                row_data.append(f"{t:.0f} & {T:.1f} & {eT:.3f}")
            else:
                row_data.append(" & & ")
        print("    " + " & ".join(row_data) + " \\\\")
    
    print("    \\bottomrule")
    print("    \\end{tabular}")
    print("    \\label{tab:A3}")
    print("\\end{table}")

#print_table(data, error, 2)

T_m = 291.9
#T_m = 289.9
dT_m = 0.2

T_Eis = 0 + 273.15
T_w = 31 + 273.15
dT_w = 0.001*31 + 0.2

m_Eis = 0.05007
#m_Eis = 0.04952
dm_Eis = dm_w
m_w = 0.34462
#m_w = 0.29517
m = m_w + m_Ä
dm = np.sqrt(dm_w**2 + dm_Ä**2)

Gamma_Eis = c_w * ((m / m_Eis) * (T_w - T_m) - (T_m - T_Eis))

d_Gamma_Eis = np.sqrt(
    ((Gamma_Eis * dc_w / c_w) ** 2) +
    ((c_w * (T_w - T_m) * dm / m_Eis) ** 2) +
    ((c_w * m * (T_w - T_m) * dm_Eis / (m_Eis ** 2)) ** 2) +
    ((c_w * m * dT_w / m_Eis) ** 2) +
    ((c_w * (m / m_Eis + 1) * dT_m) ** 2)
)
print()
print(f"$m_{{Eis}} = ({m_Eis:.6f} \\pm {dm_Eis:.6f})$ kg")
print(f"$m = ({m:.6f} \\pm {dm:.6f})$ kg")
print(f"$m_{{w}} = {m_w:.6f} \\pm {dm_w:.6f}$ kg")
print()
print(f"$T_m = ({T_m:.4f} \\pm {dT_m:.4f})$ K")
print(f"$T_{{Eis}} = ({T_Eis:.4f} \\pm {0:.4f})$ K")
print(f"$T_w = ({T_w:.4f} \\pm {dT_w:.4f})$ K")


print()
print(f"$\Gamma_{{Eis}} = ({Gamma_Eis / 1000:.6f} \pm {d_Gamma_Eis / 1000:.6f})$ kJ/(kg·K)")


#plt.xlabel('Time (s)')
#plt.ylabel('Temperature (°C)')
#plt.title('Temperature vs Time')
#plt.errorbar(data['t'], data['T'], color='#51b2e2', linestyle='dotted', marker='x', yerr=error, capsize=5)
#plt.show()