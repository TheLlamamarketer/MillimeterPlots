import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Split the line by commas and strip whitespace
            columns = [col.strip() for col in line.split(',')]

            # Ensure there are at least 6 columns
            if len(columns) >= 6:
                # Extract the last three columns
                last_three_columns = columns[4:7]

                # Join and write to output file
                outfile.write(", ".join(last_three_columns) + "\n")


# File paths (update as needed)
input_file = "test4.txt"  # Input file with raw data
output_file = "out_test4.txt"  # Output file with processed data
import numpy as np

x = np.linspace(0, 90, 1000) 
sin_x = np.abs(np.sin(x * np.pi / 180)) 
linear_x = np.abs(x)*(np.pi / 180) 

absolute_diff = np.abs(sin_x - linear_x)
divergence_angle = x[np.where(absolute_diff > 0.01)[0][0]]
fit_limit = x[x <= divergence_angle]
#print(divergence_angle )



import matplotlib.pyplot as plt
import pandas as pd
from Functions.help import *
from Functions.plotting import plot_data, DatasetSpec

data0 = {
    "l":[ 68, 62, 56, 49, 43, 37, 34, 30, 25, 20, 16, 12, 8 , 3 , 0 ],
    "alpha":[ 6.0, 5.0, 4.5, 3.5, 3.0, 2.5, 2.5, 2.0, 2.0, 1.5, 1.0, 1.0, 0.5, 0.0, 0.0 ]
}

data0["l"] = [val/100 for val in data0["l"]]

result0 = lmfit(data0["l"], data0["alpha"], yerr=0.5)
params0 = extract_params(result0)

print(f"Steigung: {round_val(params0['b'][0], params0['b'][1], False)[0]} \\pm {round_val(params0['b'][0], params0['b'][1], False)[1]}")

x = np.linspace(min(data0["l"]), max(data0["l"]), 300)
datasets = [
    {
        "xdata": data0["l"],
        "ydata": data0["alpha"],
        "y_error": 1,
        "x_error": 0.01,
        "marker": ".",
        "label": "Datenpunkte",
        "confidence": calc_CI(result0, x),
        "fit": lambda x: params0['a'][0] + params0['b'][0] * x,
        "line_fit": "-",
    },
    {},
]

plot_data(
    filename="Plots/POL_0_Paul.pdf",
    datasets=datasets,
    xlabel="Länge l/dm",
    ylabel="Winkel α/°",
    title="Polarimeter Daten",
    color_seed=25,
    plot=False
)

data = {"dAngle": 1}
raw_data = {
    "0": {
        "Angle": np.array([-90, -85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]),
        "light": np.array([779, 786, 779, 759, 725, 683, 633, 575, 517, 460, 401, 348, 297, 251, 210, 172, 146, 128, 121, 125, 139, 163, 196, 240, 284, 334, 388, 445, 502, 565, 620, 675, 718, 750, 774, 785, 785])
    },
    "1": {
        "Angle": np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]),
        "light": np.array([486, 426, 363, 300, 245, 198, 163, 137, 125, 124, 135, 160, 195, 238, 288, 346, 405, 481]),
    },
    "2": {
        "Angle": np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35]),
        "light": np.array([518, 454, 380, 309, 248, 195, 158, 133, 126, 131, 152, 187, 230, 289, 350, 427, 499]),
    },
    "3": {
        "Angle": np.array([-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]),
        "light": np.array([490, 426, 354, 294, 239, 192, 156, 133, 124, 126, 141, 170, 212, 262, 328, 386, 459]),
    },
}

raw_data0 = {
    "1": {
        "Angle": np.array([-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]),
        "light": np.array([173, 144, 113, 86, 67, 56, 51, 53, 65, 84, 109, 140, 175, 213, 249, 295, 341, 388, 438, 473, 497, 522, 541, 549, 558, 557, 549, 539]),
    },
    "2": {
        "Angle": np.array([-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]),
        "light": np.array([188, 161, 128, 100, 77, 63, 57, 58, 67, 84, 109, 140, 178, 217, 263, 309, 352, 392, 438, 475, 505, 530, 544, 553, 563, 564, 555, 538]),
    },
    "3": {
        "Angle": np.array([-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]),
        "light": np.array([179, 161, 127, 99, 74, 58, 50, 48, 54, 68, 88, 115, 147, 181, 216, 260, 298, 340, 376, 414, 447, 467, 488, 503, 508, 514, 514, 504]),
    },
}




for i in range(4):
    # Convert the hardcoded data to a DataFrame to match the second code's format
    data_i = pd.DataFrame(raw_data[str(i)])
    data[str(i)] = {
        "Angle": -data_i["Angle"],
        "dlight": 2,  
        "light": data_i["light"],
        "results": [], "params": [], "x": [], "y": [], "dx": [], "x2": [], "y2": [], "dx2": [],
        "CI": [], "R2": [],
        "results0": [], "params0": [], "Angle0": [], "light0": [],
        "alpha": [], "dalpha": []
    }

    dataN = data[str(i)]
    dataN["light"] -= dataN["light"].min()
    zero = dataN["light"][dataN["light"] == 0].index

    zero_angle = dataN["Angle"][zero].values[0]
    dataN['Angle0'] = dataN["Angle"][(dataN["Angle"] <= 22 + zero_angle) & (dataN["Angle"] >= -22 + zero_angle)]
    index0 = np.isin(dataN["Angle"], dataN["Angle0"])
    dataN["light0"] = dataN["light"][index0]
    dataN['results0'] = lmfit(dataN["Angle0"], dataN["light0"], model="quadratic")
    dataN['params0'] = extract_params(dataN['results0'])

    print("Fit0:", -dataN["params0"]['b'][0] / (2 * dataN["params0"]['c'][0]),
          -dataN["params0"]['b'][0] / (2 * dataN["params0"]['c'][0]) * np.sqrt((dataN["params0"]['b'][1] / dataN["params0"]['b'][0])**2 + (dataN["params0"]['c'][1] / dataN["params0"]['c'][0])**2),
          calc_R2(dataN["results0"]))

    sorted_indices = np.argsort(dataN["Angle"])
    dataN["y"] = dataN["Angle"].iloc[sorted_indices].to_numpy()
    dataN["x"] = np.sqrt(dataN["light"].iloc[sorted_indices]).to_numpy()
    dataN["dx"] = (dataN["dlight"] / np.sqrt(dataN["x"]) / 2)
    dy = data["dAngle"]

    dataN["x"][:np.where(dataN["x"] == dataN["x"].min())[0][0]] *= -1
    results1 = lmfit(dataN["x"], dataN["y"], dy)
    params1 = extract_params(results1)

    dataN["y2"] = dataN["y"][(dataN["y"] <= 22 + params1['a'][0]) & (dataN["y"] >= -22 + params1['a'][0])]
    index2 = np.isin(dataN["y"], dataN["y2"])
    dataN["x2"] = dataN["x"][index2]
    dataN["dx2"] = dataN["dx"][index2]

    index_min2 = np.where(dataN["x2"] == dataN["x2"].min())[0][0]
    dataN["x2"][:index_min2] *= -1

    dataN["results"] = lmfit(dataN["x2"], dataN["y2"], yerr = dataN["dlight"])
    dataN["params"] = extract_params(dataN["results"])

    print("Fit:", dataN["params"]['a'][0], dataN["params"]['a'][1], calc_R2(dataN["results"]))
    dataN["alpha"], dataN["dalpha"] = dataN["params"]['a']



max_val = [max(data[str(i)]['y']) for i in range(4)]
min_val = [min(data[str(i)]['y']) for i in range(4)]
max_val_volt = [max(data[str(i)]['light']) for i in range(4)]
min_val_volt = [min(data[str(i)]['light']) for i in range(4)]


datasets = []
for i in range(4):  # Adjust the range as needed
    x = np.linspace(data[str(i)]["Angle"].min(), data[str(i)]["Angle"].max(), 300)

    params = data[str(i)]["params0"]
    a, b, c = params['a'][0], params['b'][0], params['c'][0]

    datasets.append({
        "xdata": data[str(i)]["Angle"],
        "ydata": data[str(i)]["light"],
        "label": f"Messreie {i}",
        "marker": ".",
        "color_group": f"{i}",
        "x_error": data['dAngle'],
        "y_error": data[str(i)]["dlight"],
        "confidence": calc_CI(data[str(i)]["results0"], x),
        "fit": (lambda x, a=a, b=b, c=c: a + b*x + c*x**2),
    })

plot_data(
    filename="Plots/POL_1_Paul.pdf",
    datasets=datasets,
    xlabel="Winkel γ/°", 
    ylabel="Licht E/lux mit Offset",
    title="Polarimeter Daten",
    ymax=max(max_val_volt)+5,
    ymin=min(min_val_volt)-5,
    color_seed=54,
    plot=False
)



max_val_volt2 = [max(data[str(i)]['light']) for i in range(4)]
min_val_volt2 = [min(data[str(i)]['light']) for i in range(4)]


datasets = [{}]
for i in range(4):  # Adjust the range as needed
    x = np.linspace(data[str(i)]["Angle"].min(), data[str(i)]["Angle"].max(), 300)
    params = data[str(i)]["params0"]
    a, b, c = params['a'][0], params['b'][0], params['c'][0]

    y = (lambda x, a=a, b=b, c=c: a + b*x + c*x**2)

    confidence = {
        key: (
            np.array(group[0]) - y(x),  # Lower bound
            np.array(group[1]) - y(x)   # Upper bound
        )
        for key, group in calc_CI(data[str(i)]["results0"], x).items()
    }

    datasets.append({
        "xdata": data[str(i)]["Angle"],
        "ydata": data[str(i)]["light"] - y(data[str(i)]["Angle"]),
        "label": f"Messreie {i}",
        "marker": ".",
        "color_group": f"{i}",
        "x_error": data['dAngle'],
        "y_error": data[str(i)]["dlight"],
        "confidence":confidence,
    })

plot_data(
    filename="Plots/POL_5_Paul.pdf",
    datasets=datasets,
    xlabel="Winkel α/°", 
    ylabel="Licht E/lux mit Offset",
    title="Residuen der Polarimeter Daten",
    xmax=30,
    xmin=-45,
    ymax= 5,
    ymin=-5,
    width=20,
    height=10,
    color_seed=54,
    legend_position = None,
    plot=False
)



datasets = [
    {
        "xdata": [0,0],
        "ydata": [min(min_val), max(max_val)],
        "line": "--",
        "marker":None,
        "color": "black",
    }
]

for i in range(4):  # Adjust the range as needed
    x = np.linspace(data[str(i)]["x2"].min(), data[str(i)]["x2"].max(), 300)
    params = data[str(i)]["params"]
    a, b = params['a'][0], params['b'][0]


    datasets.append({
        "xdata": data[str(i)]["x"],
        "ydata": data[str(i)]["y"],
        "y_error": data['dAngle'],
        "line": "None",
        "marker": ".",
        "color_group": f"Fit{i}",
    })
    datasets.append({
        "xdata": data[str(i)]["x2"],
        "ydata": data[str(i)]["y2"],
        "y_error": data['dAngle'],
        "line": "None",
        "label": f"Messreihe {i}",
        "marker": ".",
        "color_group": f"Fit{i}",
        "fit": (lambda x, a=a, b=b: a + b*x),
        "confidence": calc_CI(data[str(i)]["results"], x),
    })


plot_data(
    filename="Plots/POL_2_Paul.pdf",
    datasets=datasets,
    xlabel="Wurzel der Lichtintensität $\\sqrt{E}/\\sqrt{lux}$", 
    ylabel="Winkel γ/°",
    color_seed=54,
    plot=False
)



def calculate_solution_properties(m_sucrose=None, m_water=None, m_total=None, V_solution=None):

    rho_water = 0.99705
    b = 0.0185e-3
    a = 3.6369e-3

    if m_sucrose is not None and m_water is not None:
        m_total = m_sucrose + m_water
        w = m_sucrose / m_total
    elif m_total is not None and V_solution is not None:
        w = []
        m_sucrose = []
        for i in range(0,len(m_total)):
            print(i)
            rho_solution = m_total[i] / V_solution[i]
            coeffs = [b, a, (rho_water - rho_solution)]
            roots = np.roots(coeffs)
            print(coeffs)
            print(roots)
            w_ratio = roots[roots >= 0][0] 
            m_sucrose.append(w_ratio * m_total[i])
            print(m_sucrose)
            w.append(w_ratio)

    else:
        raise ValueError("Provide either (m_sucrose and m_water) or (m_total and V_solution).")

    w = np.array(w)
    rho_solution = rho_water + a * w + b * w**2

    # Calculate final volume
    if V_solution is None:
        V_solution = m_total / rho_solution

    concentration_sucrose = m_sucrose / V_solution

    return {
        "m_sug": m_sucrose,
        "m_wat": m_total - m_sucrose,
        "m_tot": m_total,
        "V_sol": V_solution,
        "density_sol": rho_solution,
        "c_suc": concentration_sucrose,
    }



conc = 30/300
dm = 1
dV = 1
dconc = conc * np.sqrt((dm/30)**2 + (dV/300)**2)

vals0 = calculate_solution_properties(m_sucrose=30, m_water=300)
print("Conc sol 0", vals0["c_suc"])
conc0 = vals0["c_suc"]




spec_rotation = 1/conc*params0['b'][0]
spec_rotation0 = 1/conc0*params0['b'][0]
dspec_rotation = spec_rotation*np.sqrt((dconc/conc)**2 + (params0['b'][1]/params0['b'][0])**2)
dspec_rotation0 = spec_rotation0*np.sqrt((dconc/conc0)**2 + (params0['b'][1]/params0['b'][0])**2)

print(f"[\\alpha] = {round_val(spec_rotation, dspec_rotation, False)[0]} \\pm {round_val(spec_rotation, dspec_rotation, False)[1]}")
print(f"[\\alpha]_0 = {round_val(spec_rotation0, dspec_rotation0, False)[0]} \\pm {round_val(spec_rotation0, dspec_rotation0, False)[1]}")


l = 0.88
dl = 0.01
specific = 105

alphas = []
dalphas = []
for i in range(1, 4):  # Adjust the range as needed
    alphas.append(data[str(i)]["alpha"])
    dalphas.append(data[str(i)]["dalpha"])
alphas = np.array(alphas) - data['0']["alpha"]
dalphas = np.array(dalphas)

print(data['0']["alpha"])


concentration_th = alphas/l/specific
dconcentration_th = concentration_th * np.sqrt((dalphas/alphas)**2 + (dl/l)**2)

m_sug = np.array([7, 15, 30])
V_wat = np.array([300, 300, 300])
m_tot = np.array([89.06, 90.07, 97.51, 96.88])

dm = 1
dV = 1



vals2 = calculate_solution_properties(m_sucrose=m_sug, m_water=V_wat)
print("Volume sol 0", vals2["c_suc"])

c_exp_2 = vals2["c_suc"]
c_exp_3 = m_sug/V_wat
dc_exp_3 = c_exp_3 * np.sqrt((dm/m_sug)**2 + (dV/V_wat)**2)





result3 = lmfit(alphas, concentration_th, dconcentration_th, constraints={"a": 0})
params3 = extract_params(result3)


result4_2 = lmfit(alphas, c_exp_2, constraints={"a": 0})
result4_3 = lmfit(alphas, c_exp_3, constraints={"a": 0})


x = np.linspace(alphas.min(), alphas.max(), 300)
b4_2 = extract_params(result4_2)['b'][0]
b4_3 = extract_params(result4_3)['b'][0]


spec_rotation_2 = 1/l/b4_2
spec_rotation_3 = 1/l/b4_3

dspec_rotation_2 = spec_rotation_2 * np.sqrt((dl/l)**2 + (extract_params(result4_2)['b'][1]/extract_params(result4_2)['b'][0])**2)
dspec_rotation_3 = spec_rotation_3 * np.sqrt((dl/l)**2 + (extract_params(result4_3)['b'][1]/extract_params(result4_3)['b'][0])**2)

print(f"[\\alpha]_{{0}} = {round_val(spec_rotation_2, dspec_rotation_2, False)[0]} \\pm {round_val(spec_rotation_2, dspec_rotation_2, False)[1]}")
print(f"[\\alpha]_ = {round_val(spec_rotation_3, dspec_rotation_3, False)[0]} \\pm {round_val(spec_rotation_3, dspec_rotation_3, False)[1]}")

print(f"dc/d\\alpha_0 = {round_val(b4_2, extract_params(result4_2)['b'][1], False)[0]} \\pm {round_val(b4_2, extract_params(result4_2)['b'][1], False)[1]}")
print(f"dc/d\\alpha = {round_val(b4_3, extract_params(result4_3)['b'][1], False)[0]} \\pm {round_val(b4_3, extract_params(result4_3)['b'][1], False)[1]}")




print(b4_2, b4_3)

b3 = params3['b'][0]

datasets = [
    {
        "xdata": alphas,
        "ydata": concentration_th,
        "x_error": dalphas,
        "y_error": dconcentration_th,
        "line": "None",
        "marker": ".",
        "label": "Theoretisch ermittelte c",
        "fit": (lambda x, b=b3:b*x),
    },

    {
        "xdata": alphas,
        "ydata": c_exp_2,
        "x_error": dalphas,
        "line": "None",
        "marker": ".",
        "fit": (lambda x, b=b4_2:b*x),
        "label": "Experimentell ermittelte Konzentration korrigiert ",
        #"confidence": calc_CI(result4_2, x),
    },
        {
        "xdata": alphas,
        "ydata": c_exp_3,
        "x_error": dalphas,
        "y_error":dc_exp_3,
        "line": "None",
        "marker": ".",
        "fit": (lambda x, b=b4_3:b*x),
        "label": "Experimentell ermittelte Konzentration unkorrigiert",
        #"confidence": calc_CI(result4_2, x),
    },
]


plot_data(
    filename="Plots/POL_3_Paul.pdf",
    datasets=datasets,
    xlabel="Winkel α/°", 
    ylabel="concentration c/g mL^-1",
    title="Polarimeter Daten",
    color_seed=54,
    plot=True
)