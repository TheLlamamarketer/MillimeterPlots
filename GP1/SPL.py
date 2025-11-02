import numpy as np
import matplotlib.pyplot as plt

print()

data = {
    'U': [250, 240, 230, 220, 210, 200, 190, 180, 170, 160],
    'I1': [3.5, 3.443, 3.338, 3.252, 3.163, 3.072, 2.97, 2.912, 2.792, 2.7],
    'I2': [2.14, 2.1, 2.04, 1.987, 1.925, 1.864, 1.793, 1.728, 1.663, 1.59],
    'I3': [1.485, 1.44, 1.405, 1.36, 1.315, 1.27, 1.222, 1.174, 1.123, 1.065],
    'I4': [1.055, 1.062, 1.03, 0.998, 0.957, 0.918, 0.88, 0.84, 0.795, 0.75]
}

def print_table(data):
    num_rows = len(data['U'])

    print("\\begin{table}[h!]")
    print("    \\centering")
    print("    \\caption{Stromstärke I Werte für verschiedene Spannungen U und Elektronbahn Radien r}")
    print("    \\sisetup{table-format=2.2}")
    print("    \\begin{tabular}{| *{1}{S[table-format=2.2]} | *{4}{S[table-format=2.4]} |}")
    print("    \\toprule")
    header = "{$U$ (V)} & {$I_{20mm}$ (A)} & {$I_{30mm}$ (A)} & {$I_{40mm}$ (A)} & {$I_{50mm}$ (A)}"
    print(f"    {header} \\\\")
    print("    \\midrule")
    
    # Print the current (I) values with errors
    for row in range(num_rows):
        U = data['U'][row]
        I1 = data['I1'][row]
        I2 = data['I2'][row]
        I3 = data['I3'][row]
        I4 = data['I4'][row]
        
        # Use the error function directly for U and I values
        U_err = error([U], 0.008, 0.1)[0]
        I1_err = error([I1], 0.015, 0.005)[0]
        I2_err = error([I2], 0.015, 0.005)[0]
        I3_err = error([I3], 0.015, 0.005)[0]
        I4_err = error([I4], 0.015, 0.005)[0]
        
        row_data = (f"{{${U:.1f} \\pm {U_err:.1f}$}} & {{${I1:.4f} \\pm {I1_err:.4f}$}} & {{${I2:.4f} \\pm {I2_err:.4f}$}}"
                    f" &{{ ${I3:.4f} \\pm {I3_err:.4f} $}} & {{${I4:.4f} \\pm {I4_err:.4f}$}}")
        print("    " + row_data + " \\\\")
    
    # Insert a horizontal line to separate the I and I^2 values
    print("    \\midrule")
    print("    \\midrule")
    header = "{$U$ (V)} & {$I^2_{20mm} (A^2)$} & {$I^2_{30mm} (A^2)$} & {$I^2_{40mm} (A^2)$} & {$I^2_{50mm} (A^2)$}"
    print(f"    {header} \\\\")
    print("    \\midrule")

    # Print the squared current (I^2) values with errors
    for row in range(num_rows):
        U = data['U'][row]
        U_err = error([U], 0.008, 0.1)[0]
        I1_sq = data['I1'][row] ** 2
        I2_sq = data['I2'][row] ** 2
        I3_sq = data['I3'][row] ** 2
        I4_sq = data['I4'][row] ** 2

        # Use the error function directly for squared I values
        I1_sq_err = error([I1_sq], 0.015, 0.005)[0]
        I2_sq_err = error([I2_sq], 0.015, 0.005)[0]
        I3_sq_err = error([I3_sq], 0.015, 0.005)[0]
        I4_sq_err = error([I4_sq], 0.015, 0.005)[0]

        row_data_sq = (f"{{${U:.1f} \\pm {U_err:.1f}$}} &{{$ {I1_sq:.2f} \\pm {I1_sq_err:.2f}$}} & {{${I2_sq:.3f} \\pm {I2_sq_err:.3f}$}}"
                       f" & {{${I3_sq:.3f} \\pm {I3_sq_err:.3f}$}} &{{$ {I4_sq:.3f} \\pm {I4_sq_err:.3f}$}}")
        print("    " + row_data_sq + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\label{tab:1}")
    print("\\end{table}")

def error(data, percent, digit):
    return [i * percent + digit for i in data]

#print_table(data)


data = {key: [x**2 for x in value] if key.startswith('I') else value for key, value in data.items()}




def plot_data(filename, xdata, ydata, ymin, ymax, ystep, xmax, xmin, xstep, yoffset=0, xoffset=0, y_error=0, x_error=0, grey_lower=0, grey_upper=0):
    num_main_lines_x = 18
    num_main_lines_y = 28

    fig, ax = plt.subplots(figsize=(num_main_lines_x / 2.54, num_main_lines_y / 2.54))

    factors = [1, 2, 10]
    main_lines_x, secondary_lines_x, tertiary_lines_x = [
        np.arange(xmin, xmax, (xmax - xmin) / (num_main_lines_x * factor)) for factor in factors
    ]

    main_lines_y, secondary_lines_y, tertiary_lines_y = [
        np.arange(ymin, ymax, (ymax - ymin) / (num_main_lines_y * factor)) for factor in factors
    ]

    ax.set_xticks(np.arange(xmin + xoffset, xmax + 1, xstep))
    ax.set_yticks(np.arange(ymin + yoffset, ymax + 0.1, ystep))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_zorder(1)

    # tertiary grid lines
    for tick in tertiary_lines_x:
        ax.axvline(x=tick, color='#eb3107', linestyle='-', linewidth=0.1, zorder=0)
    for tick in tertiary_lines_y:
        ax.axhline(y=tick, color='#eb3107', linestyle='-', linewidth=0.1, zorder=0)

    # secondary grid lines
    for tick in secondary_lines_x:
        ax.axvline(x=tick, color='#eb3107', linestyle='-', linewidth=0.45, zorder=0)
    for tick in secondary_lines_y:
        ax.axhline(y=tick, color='#eb3107', linestyle='-', linewidth=0.45, zorder=0)

    # main grid lines
    for tick in main_lines_x:
        ax.axvline(x=tick, color='#eb3107', linestyle='-', linewidth=0.7, zorder=0)
    for tick in main_lines_y:
        ax.axhline(y=tick, color='#eb3107', linestyle='-', linewidth=0.7, zorder=0)

    for spine in ax.spines.values():
        spine.set_edgecolor('#eb3107')
        spine.set_zorder(2)

    # Plot the black points first
    plt.errorbar(xdata, ydata, color='black', marker='x', linestyle='none', yerr=y_error, xerr=x_error, capsize=5, elinewidth=1, capthick=1, clip_on=False, zorder=3)
    
    if grey_upper != 0:
        # Plot the greyed out zone
        x_array = np.array(xdata)
        y_array = np.array(ydata)
        y_error_array = np.array(y_error)
        x_error_array = np.array(x_error)
        mask_grey = (x_array >= grey_lower) & (x_array <= grey_upper)
        plt.errorbar(x_array[mask_grey], y_array[mask_grey], color='#9e9e9e', marker='x', linestyle='none', yerr=y_error_array[mask_grey], xerr=x_error_array[mask_grey] , capsize=5, elinewidth=1, capthick=1, clip_on=False, zorder=4)

    # Save the figure
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.show()

print()

def find_slope(x, y):
    slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
    intercept = np.mean(y) - slope * np.mean(x)

    slope_err = np.sqrt(np.sum((y - (slope * x + intercept))**2) / (len(x) - 2) / np.sum((x - np.mean(x))**2))
    intercept_err = slope_err * np.sqrt(1/len(x) * (1 + np.sum(x**2) / np.sum((x - np.mean(x))**2)))

    print(f"({1000* slope:.3f} \\pm {1000*slope_err:.3f})")

    return slope, slope_err

m_2, dm_2 = find_slope(np.array(data['U']), np.array(data['I1']))
m_3, dm_3 = find_slope(np.array(data['U']), np.array(data['I2']))
m_4, dm_4 = find_slope(np.array(data['U']), np.array(data['I3']))
m_5, dm_5 = find_slope(np.array(data['U'][:-1]), np.array(data['I4'][:-1]))


m_2, dm_2 = 55.6*10**(-3), 2.7*10**(-3)
m_3, dm_3 = 22.3*10**(-3), 1.2*10**(-3)
m_4, dm_4 = 12.00*10**(-3), 0.80*10**(-3)
m_5, dm_5 = 7.10*10**(-3), 0.41*10**(-3)


mu = 4 * np.pi * 10**-7
R = 0.2
dR = 0.001
N = 154.0
dN = 0.1
dr = 0.001
e_m_e = 1.758820024 * 10**11
s = (32 * N**2 * mu**2)/(125 * R**2) 
print(s)
print(e_m_e*10**(-11))


e_m_e_2 = 1/(s*m_2* 0.02**2)
e_m_e_3 = 1/(s*m_3* 0.03**2)
e_m_e_4 = 1/(s*m_4* 0.04**2)
e_m_e_5 = 1/(s*m_5* 0.05**2)

d_e_m_e_2 = e_m_e_2 * np.sqrt((dm_2/m_2)**2 + (2*dR/R)**2 + (2*dr/0.2)**2 + (2*dN/N)**2)
d_e_m_e_3 = e_m_e_3 * np.sqrt((dm_3/m_3)**2 + (2*dR/R)**2 + (2*dr/0.3)**2 + (2*dN/N)**2)
d_e_m_e_4 = e_m_e_4 * np.sqrt((dm_4/m_4)**2 + (2*dR/R)**2 + (2*dr/0.4)**2 + (2*dN/N)**2)
d_e_m_e_5 = e_m_e_5 * np.sqrt((dm_5/m_5)**2 + (2*dR/R)**2 + (2*dr/0.5)**2 + (2*dN/N)**2)


print(f"\\frac{{e}}{{m}}_{{20mm}} = ({e_m_e_2*10**(-11):.3f} \\pm {d_e_m_e_2*10**(-11):.3f})\\cdot10^{{11}}C/Kg")
print(f"\\frac{{e}}{{m}}_{{30mm}} = ({e_m_e_3*10**(-11):.3f} \\pm {d_e_m_e_3*10**(-11):.3f})\\cdot10^{{11}}C/Kg")
print(f"\\frac{{e}}{{m}}_{{40mm}} = ({e_m_e_4*10**(-11):.3f} \\pm {d_e_m_e_4*10**(-11):.3f})\\cdot10^{{11}}C/Kg")
print(f"\\frac{{e}}{{m}}_{{50mm}} = ({e_m_e_5*10**(-11):.3f} \\pm {d_e_m_e_5*10**(-11):.3f})\\cdot10^{{11}}C/Kg")


# Weighted average based on the uncertainties
def weighted_average(e_m_vals, e_m_errors):
    weights = 1 / np.array(e_m_errors)**2
    weighted_e_m = np.sum(e_m_vals * weights) / np.sum(weights)

    # Calculate the weighted variance using the formula from the image
    n = len(e_m_vals)
    
    weighted_variance = np.sum(weights * (e_m_vals - weighted_e_m)**2) / np.sum(weights)
    unbiased_factor = n / (n - 1)
    weighted_variance = weighted_variance * unbiased_factor
    
    # Standard error (error in weighted mean)
    weighted_error = np.sqrt(weighted_variance / n)
    
    return weighted_e_m, weighted_error

e_m_vals = np.array([e_m_e_2, e_m_e_3, e_m_e_4, e_m_e_5])
e_m_errors = np.array([d_e_m_e_2, d_e_m_e_3, d_e_m_e_4, d_e_m_e_5])
weighted_e_m, weighted_error = weighted_average(e_m_vals, e_m_errors)

print(f"Weighted e/m = ({weighted_e_m*10**(-11):.2f} \\pm {weighted_error*10**(-11):.2f})\\cdot10^{{11}} C/Kg")







#plot_data(ymin=7.0, ymax=12.6, xmax=250, xmin = 160, xstep=10, ystep=0.2,                                             ydata=data['I1'], xdata= data['U'], y_error=error(data['I1'], 0.015, 0.005), x_error=error(data['U'], 0.008, 0.1), filename='C:/Users/alexa/OneDrive/Desktop/newfolder/SPL1.pdf')
#plot_data(ymin=2.1, ymax=4.9,  xmax=250, xmin = 160, xstep=10, ystep=0.2, yoffset=0.1,                                ydata=data['I2'], xdata= data['U'], y_error=error(data['I2'], 0.015, 0.005), x_error=error(data['U'], 0.008, 0.1), filename='C:/Users/alexa/OneDrive/Desktop/newfolder/SPL2.pdf')
#plot_data(ymin=1.0, ymax=2.4,  xmax=250, xmin = 160, xstep=10, ystep=0.1, yoffset=0,                                  ydata=data['I3'], xdata= data['U'], y_error=error(data['I3'], 0.015, 0.005), x_error=error(data['U'], 0.008, 0.1), filename='C:/Users/alexa/OneDrive/Desktop/newfolder/SPL3.pdf')
#plot_data(ymin=0.5, ymax=1.2,  xmax=250, xmin = 160, xstep=10, ystep=0.05, yoffset=0, grey_lower=245, grey_upper=260, ydata=data['I4'], xdata= data['U'], y_error=error(data['I4'], 0.015, 0.005), x_error=error(data['U'], 0.008, 0.1), filename='C:/Users/alexa/OneDrive/Desktop/newfolder/SPL4.pdf')


