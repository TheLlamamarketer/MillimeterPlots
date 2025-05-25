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
input_file = "test00.txt"  # Input file with raw data
output_file = "out_test00.txt"  # Output file with processed data
import numpy as np

x = np.linspace(0, 90, 1000) 
sin_x = np.abs(np.sin(x * np.pi / 180))  # |sin(x * pi/180)|
linear_x = np.abs(x)*(np.pi / 180) 

absolute_diff = np.abs(sin_x - linear_x)
divergence_angle = x[np.where(absolute_diff > 0.01)[0][0]]
fit_limit = x[x <= divergence_angle]
#print(divergence_angle )



import matplotlib.pyplot as plt
import pandas as pd
from help import *
from plotting import plot_data
from tables import print_standard_table

data0 = {
    "l":[0, 0.55, 1.1, 1.65, 2.2, 2.75, 3.3, 3.85, 4.4, 4.95, 5.5, 6.05, 6.6, 7.15, 7.7, 8.25, 8.8, 9.35, 9.9, 10.45, 10.6],
    "angle_sensor": [-1, 0, 1, 1, 1.5, 2, 2.5, 2.5, 3, 4, 4, 4.5, 5.5, 5.5, 6.5, 6.5, 7, 7.5, 8, 9, 8.5 ],
    "angle_eye": [0, 0, 0, 1, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 6.5, 7, 7.5, 8, 8.5],
    "dl":[]
}

# add the error that constantly increases

data0["dl"] = [np.sqrt((0.005)**2 * i) for i in range(1, 22)]


conc = 8/84.5
dm = 0.05
dV = 0.5
dconc = conc * np.sqrt((dm/8)**2 + (dV/84.5)**2)

data0["l"] = [val/10 for val in data0["l"]]

print(f"concetration: {round_val(conc, dconc, False)[0]} \pm {round_val(conc, dconc, False)[1]}")


result0_sensor = lmfit(data0["l"], data0["angle_sensor"], yerr=0.5)
params0_sensor = extract_params(result0_sensor)

result0_eye = lmfit(data0["l"], data0["angle_eye"], yerr=0.5)
params0_eye = extract_params(result0_eye)



x = np.linspace(min(data0["l"]), max(data0["l"]), 300)
datasets = [
    {
        "xdata": data0["l"],
        "ydata": data0["angle_eye"],
        "y_error": 0.5,
        "x_error": data0["dl"],
        "marker": ".",
        "label": "Auge Daten",
        "color_group": "eye",
        "confidence": calc_CI(result0_eye, x),
        "fit": lambda x: params0_eye['a'][0] + params0_eye['b'][0] * x,
        "line_fit": "-",
    },
    {
        "xdata": data0["l"],
        "ydata": data0["angle_sensor"],
        "y_error": 0.5,
        "x_error": data0["dl"],
        "marker": ".",
        "label": "Sensor Daten",
        "color_group": "sensor",
        "confidence": calc_CI(result0_sensor, x),
        "fit": lambda x: params0_sensor['a'][0] + params0_sensor['b'][0] * x,
        "line_fit": "-",
    }
] 

plot_data(
    filename="Plots/POL_0.pdf",
    datasets=datasets,
    x_label="Weg l/dm ",
    y_label="Winkel α/°",
    title="Polarimeter Daten",
    color_seed=54,
    plot=False
)



headers = {
    "l": {"label": "{$l$ (dm)}", "err":data0["dl"]},      
    "angle_sensor": {"label": "{$\\alpha_{sensor} (°)$}", "err": 0.5},
    "angle_eye": {"label": "{$\\alpha_{auge} (°)$}", "err": 0.5},
}

print_standard_table(
    data=data0,
    headers=headers,
    column_formats= ["2.1"] * len(headers),
    caption="Daten Polarimeter mit veränderbarer Länge $l$",
    label="tab:A2",
    show=False
)



spec_rotation_sensor = 1/conc*params0_sensor['b'][0]
dspec_rotation_sensor = spec_rotation_sensor*np.sqrt((dconc/conc)**2 + (params0_sensor['b'][1]/params0_sensor['b'][0])**2)

spec_rotation_eye = 1/conc*params0_eye['b'][0]
dspec_rotation_eye = spec_rotation_eye*np.sqrt((dconc/conc)**2 + (params0_eye['b'][1]/params0_eye['b'][0])**2)

print(f"d\\alpha/dl_{{Sensor}} = {round_val(params0_sensor['b'][0], params0_sensor['b'][1], False)[0]} \pm {round_val(params0_sensor['b'][0], params0_sensor['b'][1], False)[1]}")
print(f"d\\alpha/dl_{{Auge}} = {round_val(params0_eye['b'][0], params0_eye['b'][1], False)[0]} \pm {round_val(params0_eye['b'][0], params0_eye['b'][1], False)[1]}")


print(f"[\\alpha]_{{Sensor}} = {round_val(spec_rotation_sensor, dspec_rotation_sensor, False)[0]} \\pm {round_val(spec_rotation_sensor, dspec_rotation_sensor, False)[1]}")
print(f"[\\alpha]_{{Auge}} = {round_val(spec_rotation_eye, dspec_rotation_eye, False)[0]} \\pm {round_val(spec_rotation_eye, dspec_rotation_eye, False)[1]}")

data = {"dAngle": 0.5}

for i in range(6):
    data_i = pd.read_csv(f"out_test{i}.txt", header=None, names=["Angle", "dVolts", "Volts"])
    data[str(i)] = {
        "Angle": data_i["Angle"] if i == 0 or i == 1 else data_i["Angle"] * 2,
        "dVolts": data_i["dVolts"],
        "Volts": data_i["Volts"],
        "results": [], "params": [], "x": [], "y": [], "dx": [], "x2": [], "y2": [], "dx2": [],
        "CI": [], "R2": [], 
        "results0": [], "params0": [], "Angle0": [], "Volts0": [],
        "alpha": [], "dalpha": []
    }

    dataN = data[str(i)]
    zero_voltage_indices = dataN["Angle"][dataN["Angle"] == 0].index
    offset = dataN["Volts"].iloc[zero_voltage_indices[0]] - dataN["Volts"].iloc[zero_voltage_indices[1]]
    dataN["Volts"].iloc[zero_voltage_indices[1]:] += offset
    dataN["Volts"] -= dataN["Volts"].min()
    zero = dataN["Volts"][dataN["Volts"] == 0].index

    zero_angle = dataN["Angle"][zero].values[0]
    dataN['Angle0'] = dataN["Angle"][(dataN["Angle"]<=20 + zero_angle) & (dataN["Angle"]>=-20 + zero_angle)]
    index0 = np.isin(dataN["Angle"], dataN["Angle0"])
    dataN["Volts0"] = dataN["Volts"][index0]
    dataN['results0'] = lmfit(dataN["Angle0"], dataN["Volts0"], model="quadratic")
    dataN['params0'] = extract_params(dataN['results0'])

    print(i, "Fit0:", -dataN["params0"]['b'][0]/2/dataN["params0"]['c'][0], -dataN["params0"]['b'][0]/2/dataN["params0"]['c'][0]*np.sqrt((dataN["params0"]['b'][1]/dataN["params0"]['b'][0])**2 + (dataN["params0"]['c'][1]/dataN["params0"]['c'][0])**2), calc_R2(dataN["results0"]))



    sorted_indices = np.argsort(dataN["Angle"])
    dataN["y"] = dataN["Angle"].iloc[sorted_indices].to_numpy()
    dataN["x"] = np.sqrt(dataN["Volts"].iloc[sorted_indices]).to_numpy()
    dataN["dx"] = (dataN["dVolts"].iloc[sorted_indices] / np.sqrt(dataN["x"]) / 2).to_numpy()
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

    dataN["results"] = lmfit(dataN["x2"], dataN["y2"])
    dataN["params"] = extract_params(dataN["results"])

    print("Fit:",dataN["params"]['a'][0], dataN["params"]['a'][1], calc_R2(dataN["results"]))
    dataN["alpha"], dataN["dalpha"] = dataN["params"]['a']


all_angles = sorted(
    set(data["1"]["Angle"]).union(*[data[str(i)]["Angle"] for i in range(2, 6)])
)

aligned_data = {"Angle": [angle for angle in all_angles if -40 <= angle <= 40]}

for i in range(6):
    dataset = data[str(i)]
    angle_values = dataset["Angle"]
    aligned_volts = []

    for angle in aligned_data["Angle"]:
        if angle in angle_values.values:
            idx = angle_values[angle_values == angle].index[0]
            aligned_volts.append(dataset["Volts"].iloc[idx])
        else:
            # Fill gaps with None or placeholders for missing data
            aligned_volts.append(None)

    aligned_data[f"Volts_{i}"] = aligned_volts

# Prepare headers with unified angles
headers = {
    "Angle": {"label": "{$\\gamma$ (dm)}", "err": data["dAngle"]},  
}

for i in range(1, 6):
    headers[f"Volts_{i}"] = {"label": f"{{$U_{{{i}}}$ (V)}}", "intermediate": True}

print_standard_table(
    data=aligned_data,
    headers=headers,
    column_formats=["2.1"] * len(headers),
    caption="Daten Polarimeter mit veränderbarer Winkelauflösung",
    label="tab:A3",
    show=True,
)



max_val = [max(data[str(i)]['y']) for i in range(1, 6)]
min_val = [min(data[str(i)]['y']) for i in range(1, 6)]
max_val_volt = [max(data[str(i)]['Volts']) for i in range(1, 6)]
min_val_volt = [min(data[str(i)]['Volts']) for i in range(1, 6)]


# print offset of parabola for each dataset aka the angle where the parabola reaches its minimum, aka for a + bx + cx^2 = 0, the minimum is at -b/2c


for i in range(1, 6):
    minimum_angle = -data[str(i)]["params0"]['b'][0]/2/data[str(i)]["params0"]['c'][0]
    dminimum_angle = minimum_angle*np.sqrt((data[str(i)]["params0"]['b'][1]/data[str(i)]["params0"]['b'][0])**2 + (data[str(i)]["params0"]['c'][1]/data[str(i)]["params0"]['c'][0])**2)
    print(f"Minimum {i}: {round_val(minimum_angle, dminimum_angle, False)[0]} \pm {round_val(minimum_angle, dminimum_angle, False)[1]}")


datasets = [{}]
for i in range(1,6):  # Adjust the range as needed
    x = np.linspace(data[str(i)]["Angle"].min(), data[str(i)]["Angle"].max(), 300)

    params = data[str(i)]["params0"]
    a, b, c = params['a'][0], params['b'][0], params['c'][0]

    datasets.append({
        "xdata": data[str(i)]["Angle"],
        "ydata": data[str(i)]["Volts"],
        "label": f"Messreie {i}",
        "marker": ".",
        "color_group": f"{i}",
        "x_error": data['dAngle'],
        "y_error": data[str(i)]["dVolts"],
        "confidence": calc_CI(data[str(i)]["results0"], x),
        "fit": (lambda x, a=a, b=b, c=c: a + b*x + c*x**2),
    })


plot_data(
    filename="Plots/POL_1.pdf",
    datasets=datasets,
    x_label="Winkel γ/°", 
    y_label="Spannung U/V mit Offset",
    title="Polarimeter Daten",
    ymax=max(max_val_volt)+0.01,
    ymin=min(min_val_volt)-0.01,
    color_seed=54,
    plot=False
)

max_val_volt2 = [max(data[str(i)]['Volts']) for i in range(1, 6)]
min_val_volt2 = [min(data[str(i)]['Volts']) for i in range(1, 6)]


datasets = [{}]
for i in range(1,6):  # Adjust the range as needed
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
        "ydata": data[str(i)]["Volts"] - y(data[str(i)]["Angle"]),
        "label": f"Messreie {i}",
        "marker": ".",
        "color_group": f"{i}",
        "x_error": data['dAngle'],
        "y_error": data[str(i)]["dVolts"],
        "confidence":confidence,
    })

plot_data(
    filename="Plots/POL_5.pdf",
    datasets=datasets,
    x_label="Winkel α/°", 
    y_label="Spannung U/V mit Offset",
    title="Polarimeter Daten",
    xmax=45,
    xmin=-30,
    ymax= 0.1,
    ymin=-0.4,
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

for i in range(1, 6):
    print(f"Minimum {i}: {round_val(data[str(i)]["params"]['a'][0], data[str(i)]["params"]['a'][1], False)[0]} \pm {round_val(data[str(i)]["params"]['a'][0], data[str(i)]["params"]['a'][1], False)[1]}")


for i in range(1, 6):  # Adjust the range as needed
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
    filename="Plots/POL_2.pdf",
    datasets=datasets,
    x_label="Wurzel der Spannung √U/√V", 
    y_label="Winkel γ/°",
    title="Polarimeter Daten",
    color_seed=54,
    plot=False
)



def calculate_solution_properties(m_sucrose=None, m_water=None, m_total=None, V_solution=None, dm=None, dV=None):

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
            rho_solution = m_total[i] / V_solution[i]
            coeffs = [b, a, (rho_water - rho_solution)]
            roots = np.roots(coeffs)
            w_ratio = roots[roots >= 0][0] 
            m_sucrose.append(w_ratio * m_total[i])
            w.append(w_ratio)

    else:
        raise ValueError("Provide either (m_sucrose and m_water) or (m_total and V_solution).")

    w = np.array(w)
    rho_solution = rho_water + a * w + b * w**2

    # Calculate final volume
    if V_solution is None:
        V_solution = m_total / rho_solution

    concentration_sucrose = m_sucrose / V_solution

    # Error propagation
    dm_sucrose = dm if dm is not None else np.zeros_like(m_sucrose)
    dV_solution = (
        V_solution
        * np.sqrt(
            (dm / m_total) ** 2
            + ((a + 2 * b * w) * dm / rho_solution) ** 2
        )
        if dV is not None
        else np.zeros_like(V_solution)
    )

    dc_sucrose = concentration_sucrose * np.sqrt(
        (dm_sucrose / m_sucrose) ** 2 + (dV_solution / V_solution) ** 2
    )

    return {
        "m_sug": m_sucrose,
        "m_wat": m_total - m_sucrose,
        "m_tot": m_total,
        "V_sol": V_solution,
        "density_sol": rho_solution,
        "c_suc": concentration_sucrose,
        "dc_suc": dc_sucrose,
    }





l = 1.06
dl = 0.005
specific = 85

alphas = []
dalphas = []
for i in range(2, 6):  # Adjust the range as needed
    alphas.append(data[str(i)]["alpha"])
    dalphas.append(data[str(i)]["dalpha"])
alphas = np.array(alphas) - data['1']["alpha"]
dalphas = np.sqrt((np.array(dalphas))**2 + (data['1']["dalpha"])**2)

alphas2 = np.array([2, 5, 8.5, 16])
dalphas2 = np.sqrt(2)*0.5

concentration_th = alphas/l/specific
dconcentration_th = concentration_th * np.sqrt((dalphas/alphas)**2 + (dl/l)**2)

m_sug = np.array([2.05, 4.3, 8.92, 16])
V_wat = np.array([83, 86, 89, 81])
m_tot = np.array([89.06, 90.07, 97.51, 96.88])
V_tot = np.array([88, 88, 94, 91])

dm = 0.05
dV = 0.5



c_exp_1 = m_sug/V_tot
dc_exp_1 = c_exp_1 * np.sqrt((dm/m_sug)**2 + (dV/V_tot)**2)
vals2 = calculate_solution_properties(m_sucrose=m_sug, m_water=V_wat, dm=dm, dV=dV)
c_exp_2 = vals2["c_suc"]
dc_exp_2 = vals2["dc_suc"]





result3 = lmfit(alphas, concentration_th, dconcentration_th, constraints={"a": 0})
params3 = extract_params(result3)

result4_0 = lmfit(alphas2, c_exp_1, dc_exp_1, constraints={"a": 0})
result4_1 = lmfit(alphas, c_exp_1, dc_exp_1, constraints={"a": 0})
result4_2 = lmfit(alphas, c_exp_2, np.array(dc_exp_2), constraints={"a": 0})


x = np.linspace(alphas.min(), alphas.max(), 300)
b4_0 = extract_params(result4_0)['b'][0]
b4_1 = extract_params(result4_1)['b'][0]
b4_2 = extract_params(result4_2)['b'][0]




print(b4_1, b4_2)
b3 = params3['b'][0]


spec_rotation_0 = 1/l/b4_0
spec_rotation_1 = 1/l/b4_1
spec_rotation_2 = 1/l/b4_2

dspec_rotation_0 = spec_rotation_0 * np.sqrt((dl/l)**2 + (extract_params(result4_0)['b'][1]/extract_params(result4_0)['b'][0])**2)
dspec_rotation_1 = spec_rotation_1 * np.sqrt((dl/l)**2 + (extract_params(result4_1)['b'][1]/extract_params(result4_1)['b'][0])**2)
dspec_rotation_2 = spec_rotation_2 * np.sqrt((dl/l)**2 + (extract_params(result4_2)['b'][1]/extract_params(result4_2)['b'][0])**2)


print(f"[\\alpha]_{{}} = {round_val(spec_rotation_0, dspec_rotation_0, False)[0]} \\pm {round_val(spec_rotation_0, dspec_rotation_0, False)[1]}")
print(f"[\\alpha]_{{Sensor}} = {round_val(spec_rotation_1, dspec_rotation_1, False)[0]} \\pm {round_val(spec_rotation_1, dspec_rotation_1, False)[1]}")
print(f"[\\alpha]_{{Sensor}} = {round_val(spec_rotation_2, dspec_rotation_2, False)[0]} \\pm {round_val(spec_rotation_2, dspec_rotation_2, False)[1]}")

# print R2 for each of these fits
print(calc_R2(result4_0))
print(calc_R2(result4_1))
print(calc_R2(result4_2))




headers = {
    "m_sug": {"label": "{$m_{zucker}$ (g)}", "err":dm, "data": m_sug},
    "m_tot": {"label": "{$m_{gesamt}$ (g)}", "err":dm, "data": m_tot},
    "V_sol": {"label": "{$V_{gesamt}$ (ml)}", "err":dV, "data": V_tot},
    "conc": {"label": "{$c_{exp}$ (g/ml)}", "err":dc_exp_1, "data": c_exp_1},
    "alpha2":{ "label": "{$\\alpha_{manuell}$ (°)}", "err":dalphas2, "data": alphas2},
    "alpha": {"label": "{$\\alpha_{numerisch}$ (°)}", "err":dalphas, "data": alphas},
}



print_standard_table(
    data=data,
    headers=headers,
    column_formats= ["2.1"] * len(headers),
    caption="Daten Polarimeter mit veränderbarer Konzentration",
    label="tab:A3_2",
    show=True
)

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
        "ydata": c_exp_1,
        "x_error": dalphas,
        "y_error":dc_exp_1,
        "line": "None",
        "marker": ".",
        "fit": (lambda x, b=b4_1:b*x),
        "label": "Experimentell ermittelte Konzentration m/V",
        #"confidence": calc_CI(result4_1, x),
    },
    {
        "xdata": alphas2,
        "ydata": c_exp_1,
        "x_error": dalphas2,
        "y_error":dc_exp_1,
        "line": "None",
        "marker": ".",
        "fit": (lambda x, b=b4_0:b*x),
        "label": "Experimentell ermittelte Konzentration m/V mit manuellen Winkeln",
        #"confidence": calc_CI(result4_1, x),
    },
    {
        "xdata": alphas,
        "ydata": c_exp_2,
        "x_error": dalphas,
        "y_error":dc_exp_2,
        "line": "None",
        "marker": ".",
        "fit": (lambda x, b=b4_2:b*x),
        "label": "Experimentell ermittelte Konzentration mit korrigierten Werten",
        #"confidence": calc_CI(result4_2, x),
    },
]


plot_data(
    filename="Plots/POL_3.pdf",
    datasets=datasets,
    x_label="Winkel α/°", 
    y_label="concentration c/g mL^-1",
    title="Polarimeter Daten",
    color_seed=54,
    plot=False
)


