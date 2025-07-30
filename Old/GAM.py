import numpy as np
import sys
sys.path.append('../Millimeterplots')
from Old.plotting_old import plot_data
from find_interval import find_interval

print()

data = {
    "a": [3048, 3436, 1786, 1377, 3301, 80.2, 181],
    "b": [62.8, 63.1, 46.3, 42.1, 60.7, 28.2, 8.33],
    "E": [1.173, 1.333, 0.662, 0.511, 1.275, 0.027, 0.056],
}

# find the slope through linear regression
slope = np.sum((np.array(data["a"]) - np.mean(data["a"]))* (np.array(data["E"]) - np.mean(data["E"]))) / np.sum((np.array(data["E"]) - np.mean(data["E"])) ** 2)
intercept = np.mean(data["a"]) - slope * np.mean(data["E"])

d_slope = np.sqrt( np.sum((np.array(data["a"]) - slope * np.array(data["E"]) - intercept) ** 2) / (len(data["a"]) - 2) / np.sum((np.array(data["E"]) - np.mean(data["E"])) ** 2) )
d_intercept = d_slope * np.sqrt(np.sum(np.array(data["E"]) ** 2) / len(data["E"]))

slope = 2554
d_slope = 77
intercept = 49
d_intercept = 38


def E(N):
    return (N - intercept) / slope


def d_E(N):
    return np.sqrt(
        (d_intercept / slope) ** 2 + ((N - intercept) / slope**2 * d_slope) ** 2
    )


print(f"$E = {E(1377):.3f} \\pm {d_E(1377):.3f}$")

print(
    f"Slope: {slope:.1f} \\pm {d_slope:.1f}, Intercept: {intercept:.1f} \\pm {d_intercept:.1f}"
)


def print_table(data):
    num_rows = len(data["a"])

    print("\\begin{table}[h!]")
    print("    \\centering")
    print(
        "    \\caption{{Messdaten zur Analyse verschiedener Isotope ($a$ sind die gemessenen Werte der Intensität, $b$ die Abweichungen und $E$ die Energie in MeV)}}"
    )
    print("    \\sisetup{table-format=4.1}")
    print(
        "    \\begin{tabular}{| *{1}{S[table-format=4.1]} | *{3}{S[table-format=4.1]} |}"
    )
    print("    \\toprule")
    header = "{Peak} & {$a$(\\#)} & {$b$(\\#)} & {E (MeV)}"
    print(f"    {header} \\\\")
    print("    \\midrule")

    peaks = ["CO 1", "CO 2", "CS 1", "Na 1", "Na 2", "Am 1", "Am 2"]
    for row in range(num_rows):
        a = data["a"][row]
        b = data["b"][row]
        E = data["E"][row]
        row_data = f"{peaks[row]} & {a:.1f} & {b:.1f} & {E:.3f}"
        print("    " + row_data + " \\\\")

    print("    \\bottomrule")
    print("    \\end{tabular}")
    print("    \\label{tab:IsotopAnalyse}")
    print("\\end{table}")


print_table(data)

# find_interval(data['E'], data['a'], name='data', max_increment_height=500, max_increment_width=1)

plot_data(ymin=0, ymax=3640, xmin=0, xmax=1.4, xstep=0.14, ystep=130, x_label='E/MeV', y_label='N/#', datasets=[{'xdata':data['E'], 'ydata': data['a'], 'y_error': data['b']}], filename='Plots/GAM_1.pdf')

data_pb = {
    "d": [3, 6, 9, 15, 18, 12, 15],
    "A": [12296, 9133, 6968, 3502, 2471, 3502, 2471],
}

data_fe = {
    "d": [9.9, 19.9, 30, 35],
    "A": [32365, 13409, 6139, 4310],
}

grey_zones = [
    {"x_val": 12, "y_val": np.log(3502)},
    {"x_val": 15, "y_val": np.log(2471)},
]

data_pb["A"] = data_pb["A"][:5]
data_pb["d"] = data_pb["d"][:5]


def print_table(data_pb, data_fe):
    num_rows = len(data_pb["d"])

    print("\\begin{table}[h!]")
    print("    \\centering")
    print("    \\caption{{Ja)}}")
    print("    \\sisetup{table-format=2.2}")
    print(
        "    \\begin{tabular}{| *{1}{S[table-format=2.1]} *{1}{S[table-format=5.0]} *{1}{S[table-format=2.1]} | *{1}{S[table-format=2.1]} *{1}{S[table-format=5.0]} *{1}{S[table-format=2.1]} |}"
    )
    print("    \\toprule")
    print("    \\multicolumn{3}{|c|}{Blei} & \\multicolumn{3}{c|}{Eisen} \\\\")
    print("    \\midrule")
    header = "{d} & {I} & {ln(I)}"
    print(f"    {header} & {header} \\\\")
    print("    \\midrule")

    for row in range(num_rows):
        d_pb = data_pb["d"][row]
        A_pb = data_pb["A"][row]
        if row >= len(data_fe["d"]):
            row_data = f"{d_pb:.1f} & {A_pb:.1f} & {np.log(A_pb):.1f} & & &"
            print("    " + row_data + " \\\\")
            continue
        else:
            d_fe = data_fe["d"][row]
            A_fe = data_fe["A"][row]
            row_data = f"{d_pb:.1f} & {A_pb:.1f} & {np.log(A_pb):.1f} & {d_fe:.1f} & {A_fe:.1f} & {np.log(A_fe):.1f}"
            print("    " + row_data + " \\\\")

    print("    \\bottomrule")
    print("    \\end{tabular}")
    print("    \\label{tab:Dick}")
    print("\\end{table}")


# print_table(data_pb, data_fe)


data_pb = {
    key: [np.log(x) for x in value] if key.startswith("A") else value
    for key, value in data_pb.items()
}
data_fe = {
    key: [np.log(x) for x in value] if key.startswith("A") else value
    for key, value in data_fe.items()
}


print(data_pb["d"])

slope_pb = np.sum(
    (np.array(data_pb["A"]) - np.mean(data_pb["A"]))
    * (np.array(data_pb["d"]) - np.mean(data_pb["d"]))
) / np.sum((np.array(data_pb["d"]) - np.mean(data_pb["d"])) ** 2)
slope_fe = np.sum(
    (np.array(data_fe["A"]) - np.mean(data_fe["A"]))
    * (np.array(data_fe["d"]) - np.mean(data_fe["d"]))
) / np.sum((np.array(data_fe["d"]) - np.mean(data_fe["d"])) ** 2)

intercept_pb = np.mean(data_pb["A"][:-2]) - slope_pb * np.mean(data_pb["d"][:-2])
intercept_fe = np.mean(data_fe["A"]) - slope_fe * np.mean(data_fe["d"])

d_slope_pb = np.sqrt(
    np.sum(
        (np.array(data_pb["A"]) - slope_pb * np.array(data_pb["d"]) - intercept_pb) ** 2
    )
    / (len(data_pb["A"]) - 2)
    / np.sum((np.array(data_pb["d"]) - np.mean(data_pb["d"])) ** 2)
)
d_slope_fe = np.sqrt(
    np.sum(
        (np.array(data_fe["A"]) - slope_fe * np.array(data_fe["d"]) - intercept_fe) ** 2
    )
    / (len(data_fe["A"]) - 2)
    / np.sum((np.array(data_fe["d"]) - np.mean(data_fe["d"])) ** 2)
)

slope_pb = -110.9
slope_fe = -84.2

print(f"$\\mu_{{Pb}}= ({slope_pb:.1f})/m$")
print(f"$\\mu_{{Fe}}= ({slope_fe:.1f})/m$")

print(f"$\\mu_{{Pb}}/\\rho_{{Pb}} = ({slope_pb/11.3:.4f}) m^2/kg$")
print(f"$\\mu_{{Fe}}/\\rho_{{Fe}} = ({slope_fe/7.8:.4f}) m^2/kg$")

# half depth of penetration
print(f"$d_{{1/2,Pb}} = ({np.log(2)/slope_pb*10**3:.4f})mm$")
print(f"$d_{{1/2,Fe}} = ({np.log(2)/slope_fe*10**3:.4f})mm$")

# find_interval(data_pb['d'], data_pb['A'], name='data_pb', max_increment_height=1, max_increment_width=5)
# find_interval(data_fe['d'], data_fe['A'], name='data_fe', max_increment_height=1, max_increment_width=5)

# find_interval([3, 35], [7.75, 10.5], name='data_combi', max_increment_height=2, max_increment_width=5)

# plot_data(ymin=7.75, ymax=9.5, xmin=2.4, xmax=18.4, xstep=1.6, ystep=0.25, xtickoffset=0, grey_zones=grey_zones, datasets=[{'xdata': data_pb['d'], 'ydata': data_pb['A'], 'x_error': [0.1]*len(data_pb['d']), 'color': '#003366'}], filename='Plots/GAM_2.pdf')
# plot_data(ymin=8, ymax=10.8, xmin=7, xmax=37, xstep=3, ystep=0.1, datasets=[{'xdata': data_fe['d'], 'ydata': data_fe['A'], 'x_error': [0.1]*len(data_fe['d']), 'color': '#006400'}], filename='Plots/GAM_3.pdf')

# plot_data(
#    ymin=7.7,
#    ymax=10.5,
#    ystep=0.1,
#    xmax=35,
#    xmin=3,
#    xstep=2,
#    grey_zones=grey_zones,
#    x_label="d/mm",
#    y_label="I/ #/s",
#    filename="Plots/GAM_4.pdf",
#    datasets=[
#        {
#            "xdata": data_pb["d"],
#            "ydata": data_pb["A"],
#            "x_error": [0.1] * len(data_pb["d"]),
#            "color": "#003366",
#            "label": "Blei",
#        },
#        {
#            "xdata": data_fe["d"],
#            "ydata": data_fe["A"],
#            "x_error": [0.1] * len(data_fe["d"]),
#            "color": "#006400",
#            "label": "Eisen",
#        },
#    ],
# )


import pandas as pd

file_path = "Old/cs2.csv"
from Old.plotting_old import generate_contrasting_color

column_names = ["Kanalnummer", "Impulse", "Fit(Impulse)"]
df = pd.read_csv(file_path, sep="\t", names=column_names)

data = {
    "Kanalnummer": df["Kanalnummer"].tolist(),
    "Impulse": df["Impulse"].tolist(),
}
# find_interval(data['Kanalnummer'], data['Impulse'], name='data', max_increment_height=100, max_increment_width=100, width=28, height=20)
# plot_data_plus(ymin=0, ymax=1000, ystep=50, xmin=0, xmax=4000, xstep=500, width=28, height=20, x_label='Kanalnummer', y_label='Impulse',marker='none', datasets=[{'xdata':data['Kanalnummer'], 'ydata': data['Impulse'], 'label':'Daten'}], filename='Plots/GAM_5.pdf')

# Extract the relevant columns into lists
data = {
    "Kanalnummer": df["Kanalnummer"][1600:2000].tolist(),
    "Impulse": df["Impulse"][1600:2000].tolist(),
    "Fit(Impulse)": df["Fit(Impulse)"][1600:2000].tolist(),
}
print((1786 - intercept) / slope)
print(110 / (1786 - intercept))

# plot_data_plus(ymin=0, ymax=980, ystep=70, xmin=1600, xmax=2000, xstep=50, x_label='Kanalnummer', y_label='Impulse', datasets=[{'xdata':data['Kanalnummer'], 'ydata': data['Impulse'], 'label':'Daten'}, {'xdata':data['Kanalnummer'], 'ydata':data['Fit(Impulse)'], 'label':'Fit'}], filename='Plots/GAM_6.pdf')


data = {
    "Kanalnummer": df["Kanalnummer"][1100:1700].tolist(),
    "Impulse": df["Impulse"][1100:1700].tolist(),
}
# find_interval(data['Kanalnummer'], data['Impulse'], name='data', max_increment_height=100, max_increment_width=100)
# plot_data_plus(ymin=0, ymax=224, ystep=20, xmin=1100, xmax=1700, xstep=50, x_label='Kanalnummer', y_label='Impulse', datasets=[{'xdata':data['Kanalnummer'], 'ydata': data['Impulse'], 'label':'Daten'}], filename='Plots/GAM_7.pdf')
