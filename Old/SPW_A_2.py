from math import sqrt
from os import error
from re import M, T
from matplotlib.pylab import f
import numpy as np
import matplotlib.pyplot as plt
from SPW_A_0 import T_w, m_Ä, dm_Ä
from SPW_Const import dm_w, dm
from SPW_A_1 import c_w, dc_w


data_fe = {
    't': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
    'T': [3.5, 5.4, 6.9, 7.4, 8.1, 8.3, 8.3, 8.4, 8.5, 8.6, 8.6, 8.6, 8.6, 8.6, 8.6]
}

data_al = {
    't': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600],
    'T': [3.2, 4.4, 5.1, 5.4, 5.6, 5.8, 5.8, 5.9, 5.9, 6, 6, 6.1, 6.1, 6.1, 6.2, 6.2, 6.2, 6.3, 6.3, 6.3, 6.4, 6.4, 6.4, 6.5, 6.5, 6.5, 6.5, 6.6, 6.6, 6.6, 6.7, 6.7, 6.7, 6.8, 6.8, 6.8, 6.8, 6.9, 6.9, 6.9, 7, 7, 7, 7, 7.1, 7.1, 7.1, 7.1, 7.2, 7.2, 7.2, 7.2, 7.3, 7.3, 7.3, 7.3, 7.4, 7.4, 7.4, 7.5]
}

error_fe = [i * 0.001 + 0.2 for i in data_fe['T']]
error_al = [i * 0.001 + 0.2 for i in data_al['T']]

def get_decimal_places(value, significant_figures=4):
    """Return the number of decimal places to represent the value with the given significant figures."""
    if value == 0:
        return 0
    decimal_places = -int(np.floor(np.log10(abs(value)))) + (significant_figures - 1)
    return max(0, decimal_places)

def format_value(value, error, significant_figures=4):
    """Format the value based on the significant figures of the error."""
    decimal_places = get_decimal_places(error, significant_figures)
    format_string = f"{{:.{decimal_places}f}}"
    return format_string.format(value)

def print_table(data, error, material, max_columns=3):

    captions = {
        "Fe": "Messdaten Aufgabe 2: Erwärmung von kaltem Wasser mit Eisenprobe bei Raumtemperatur (t ist der Zeitpunkt nach Eintauchen der Probe, T ist die Temperatur des Wassers, die das Thermometer zum Zeitpunkt t misst). In Abbildung \\ref{fig:A2_fe} grafisch dargestellt. (Messfehler berechnet in \\ref{fehler_2}).",
        "Al": "Messdaten Aufgabe 2: Erwärmung von kaltem Wasser mit Aluminiumprobe bei Raumtemperatur (t ist der Zeitpunkt nach Eintauchen der Probe, T ist die Temperatur des Wassers, die das Thermometer zum Zeitpunkt t misst). In Abbildung \\ref{fig:A2_al} grafisch dargestellt. (Messfehler berechnet in \\ref{fehler_2})"
    }
    caption = captions.get(material)

    num_rows = len(data['t'])
    rows_per_column = (num_rows + max_columns - 1) // max_columns  # Calculate the number of rows per column
    split_data = [data['t'][i:i + rows_per_column] for i in range(0, num_rows, rows_per_column)]
    split_temp = [data['T'][i:i + rows_per_column] for i in range(0, num_rows, rows_per_column)]
    split_error = [error[i:i + rows_per_column] for i in range(0, num_rows, rows_per_column)]

    print("\\begin{table}[h!]")
    print("    \\centering")
    print(f"    \\caption{{{caption}}}")
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
    print(f"    \\label{{tab:A2_{material}}}")
    print("\\end{table}")

#print_table(data_fe, error_fe, "Fe", 1)
#print_table(data_al, error_al, "Al", 3)

T_fe = 24.2 + 273.15
dT_fe = 24.2*0.001 + 0.2
T_al = 25.1 + 273.15
dT_al = 25.1 *0.001 + 0.2

T_w_fe = 2 + 273.15
dT_w_fe = 0.001*2 + 0.2
T_w_al = 0.8 + 273.15
dT_w_al = 0.001*0.8 + 0.2

m_fe = 0.83419
dm_fe = dm
m_al = 0.30622
dm_al = dm

T_m_fe = 8.3 + 273.15
dT_m_fe = 0.18
T_m_al = 278.97
dT_m_al = 0.219

m_w = 0.2345 
m = m_w + m_Ä
dM = sqrt(dm_w**2 + dm_Ä**2)

M_fe = 0.055845
M_al = 0.026982

c_mol_fe = c_w * (m * M_fe * (T_m_fe - T_w_fe)) / (m_fe * (T_fe - T_m_fe))
c_mol_al = c_w * (m * M_al * (T_m_al - T_w_al)) / (m_al * (T_al - T_m_al))


dc_mol_fe = c_mol_fe *np.sqrt(
    (dc_w / c_w) ** 2 +
    (dM / m) ** 2 +
    (dm_fe / m_fe) ** 2 +
    (dT_w_fe / (T_m_fe - T_w_fe)) ** 2 +
    (dT_fe / (T_fe - T_m_fe)) ** 2 +
    ((T_fe + T_w_fe - 2 * T_m_fe) / ((T_m_fe - T_w_fe) * (T_fe - T_m_fe)) * dT_m_fe) ** 2
)


dc_mol_al = c_mol_al *np.sqrt(
    (dc_w / c_w) ** 2 +
    (dM / m) ** 2 +
    (dm_al / m_al) ** 2 +
    (dT_w_al / (T_m_al - T_w_al)) ** 2 +
    (dT_al / (T_al - T_m_al)) ** 2 +
    ((T_al + T_w_al - 2 * T_m_al) / ((T_m_al - T_w_al) * (T_al - T_m_al)) * dT_m_al) ** 2
)

print()
print(f"$m_{{Fe}} = ({m_fe:.5f} \pm {dm_fe:.5f})$ kg")
print(f"$m_{{Al}} = ({m_al:.5f} \pm {dm_al:.5f})$ kg")
print(f"$m_W = ({m_w:.5f} \pm {dm_w:.5f})$ kg")
print()
print(f"$T_{{Fe}} = ({T_fe:.2f} \pm {dT_fe:.2f})$ K")
print(f"$T_{{Al}} = ({T_al:.2f} \pm {dT_al:.2f})$ K")
print(f"$T_{{w,Fe}} = ({T_w_fe:.2f} \pm {dT_w_fe:.2f})$ K")
print(f"$T_{{w,Al}} = ({T_w_al:.2f} \pm {dT_w_al:.2f})$ K")
print()
print(f"$T_{{m,Fe}} = ({T_m_fe:.2f} \pm {dT_m_fe:.2f})$ K")
print(f"$T_{{m,Al}} = ({T_m_al:.2f} \pm {dT_m_al:.2f})$ K")

print()
print(f"$c_{{mol,Fe}} = ({c_mol_fe:.6f} \pm {dc_mol_fe:.6f})$ kJ/(mol·K)")
print(f"$c_{{mol,Al}} = ({c_mol_al:.6f} \pm {dc_mol_al:.6f})$ kJ/(mol·K)")

#plt.xlabel('Time (s)')
#plt.ylabel('Temperature (°C)')
#plt.title('Temperature vs Time')
#plt.errorbar(data_fe['t'], data_fe['T'], label='Fe', color='#B7410E', linestyle='dotted', marker='x', yerr=error_fe, capsize=5)
#plt.errorbar(data_al['t'], data_al['T'], label='Al', color='#d0d5db', linestyle='dotted', marker='x', yerr=error_al, capsize=5)
#plt.legend()
#plt.show()