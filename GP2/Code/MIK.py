import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Functions.plotting import plot_data
from Functions.help import *
from Functions.tables import print_standard_table, print_complex_table

data = {
    "f": 4,
    "offset": 4.9,
    "div":1.33e-2,
    "ddiv":0.06e-2,
    "Ex1": {
        "beta": np.array([4, 10, 6.4, 2.9, 3.2]),
        "g_abs": np.array([19.9, 14.1, 16.3, 23.3, 22.0]),
        "b": np.array([5.1, 7.4, 6.4, 4.6, 4.7]),
        "g_rel": np.array([15.0, 9.2, 11.4, 18.4, 17.1]),
        "f": np.array([]), "df":np.array([]), 
    },
    "Ex2": {
        "x_abs": np.array([16.9, 18.3, 21.0]), "x_rel": np.array([]),
        "B": np.array([10, 10, 10]),
        "G": np.array([3, 2.5, 1.5]),
        "beta": np.array([10/3, 10/2.5, 10/1.5]), "beta2": np.array([]),
        "B2": np.array([10, 10, 50]),
        "G2": np.array([2.5, 1.5, 5]),
        "t": np.array([15, 20, 30]), 
        "dbeta":np.array([]),"dbeta2":np.array([]),
        "Gamma_exp0": np.array([]), "dGamma_exp0": np.array([]),
        "Gamma_th": np.array([]), "dGamma_th": np.array([]),
        "Gamma_exp": np.array([]),"dGamma_exp": np.array([])
    },
    "Ex3": {
        "div_width": 60/8,
        "ddiv_width": 1/8,
        "div_height": 2,
        "ddiv_height": 0.5,
    },
    "Ex4": {
        "A_1": np.array([0.2]),
        "A_2": np.array([1.0]),
        "B_1": np.array([0.3]),
        "B_2": np.array([0.6]),
        "B_3": np.array([0.4]),
    }
}


beta = data["Ex1"]["beta"]
g = data["Ex1"]["g_rel"]
b = data["Ex1"]["b"]
db = 0.1
dg = 0.2
dbeta = 0.1

data["Ex1"]["f"] = abs((g - b)/(1/beta - beta))

data["Ex1"]["df"] = np.sqrt((db/(1/beta + beta))**2 + (dg/(1/beta + beta))**2 + (dbeta*(beta**2 + 1)*(g - b)/(1 - beta**2)**2)**2)

print(data["Ex1"]["f"])
print(1/(1/b + 1/g))
print(g/b)


data["Ex2"]["x_rel"] = data["Ex2"]["x_abs"] - data["offset"]
G = data["Ex2"]["G"]
B = data["Ex2"]["B"]
data["Ex2"]["dbeta"] = 0.5*B/G**2

data["Ex2"]["dGamma_th"] = 0.1/4*(25/4 + 1)
data["Ex2"]["Gamma_exp0"] = data["Ex2"]["beta"]*(25/4 + 1)
data["Ex2"]["dGamma_exp0"] = data["Ex2"]["dbeta"]*(25/4 + 1)

print("#"*100)
print(data["Ex2"]["t"][0]/4)
print(data["Ex2"]["beta"])
print(data["Ex2"]["dbeta"])

data["Ex2"]["beta2"] = data["Ex2"]["B2"]/data["Ex2"]["G2"]
data["Ex2"]["dbeta2"] = 0.5*data["Ex2"]["B2"]/data["Ex2"]["G2"]**2

print("#"*100)
print(data["Ex2"]["t"]/4)
print(data["Ex2"]["beta2"])
print(data["Ex2"]["dbeta2"])

print("#"*100)
data["Ex2"]["Gamma_th"] = data["Ex2"]["t"]/4*(25/4 + 1)
data["Ex2"]["Gamma_exp"] = data["Ex2"]["beta2"]*(25/4 + 1)
data["Ex2"]["dGamma_exp"] = data["Ex2"]["dbeta2"]*(25/4 + 1)


print(data["Ex2"]["Gamma_th"])
print(data["Ex2"]["Gamma_exp"])
print(data["Ex2"]["dGamma_exp"])




g_abs = data["Ex1"]["g_abs"]
a0,_,b0,_,_,_ = slope(beta, g_abs -(4*(1/beta - beta) + b))
print(a0, b0)

plot_data(
    filename="Plots/MIK_2.pdf",
    datasets=[
        {
            "xdata": beta,
            "ydata": g_abs -(4*abs((1/beta - beta)) + b),
            "label": "Experimental values",
            "marker": ".",
            "line": "None",
        },
        {
            "xdata": beta,
            "ydata": a0 + b0 * beta,
            "marker": "None",
            "line": "-",
        },
    ],
    plot=False
)




a,_,b,_,_,_ = slope(data["Ex2"]["t"], data["Ex2"]["Gamma_exp"], data["Ex2"]["dGamma_exp"])

plot_data(
    filename="Plots/MIK_1.pdf",
    datasets=[
        {
            "xdata": data["Ex2"]["t"],
            "ydata": data["Ex2"]["Gamma_exp"],
            "y_error": data["Ex2"]["dGamma_exp"] ,
            "label": "Experimental values",
            "marker": ".",
        },
        {
            "xdata": data["Ex2"]["t"],
            "ydata": a + b * data["Ex2"]["t"],
            "marker": "None",
            "line": "-",
        },
        {
            "xdata": data["Ex2"]["t"],
            "ydata": data["Ex2"]["Gamma_th"],
            "label": "Theoretical values",
            "marker": ".",
            "line": "None",
        },
    ],
    plot=False
)

d_width = data["Ex3"]["div_width"] * data["div"]
d_height = data["Ex3"]["div_height"] * data["div"]
dd_width = d_width * np.sqrt((data["Ex3"]["ddiv_width"]/data["Ex3"]["div_width"])**2 + (data["ddiv"]/data["div"])**2)
dd_height = d_height * np.sqrt((data["Ex3"]["ddiv_height"]/data["Ex3"]["div_height"])**2 + (data["ddiv"]/data["div"])**2)


width, dwidth,_ = round_val(d_width, dd_width, intermed=False)
height, dheight,_ = round_val(d_height, dd_height, intermed=False)

print(f"d_{{width}} = {data['Ex3']['div_width']} ± {data['Ex3']['ddiv_width']} div")
print(f"d_{{thick}} = {data['Ex3']['div_height']} ± {data['Ex3']['ddiv_height']} div")


print(f"d_{{width}} = ({width} ± {dwidth}) mm")
print(f"d_{{thick}} = ({height} ± {dheight}) mm")



def d_min(B):
    return 550e-6 /np.sin(np.arctan(B/80))

for i, val in data["Ex4"].items():
    print(f"for {i}={val}, d_{{min}}({i})={d_min(val)}")


headers = {
    "A1": {"label": "{names}", "data": ["$A_1$", "$A_2$", "$B_1$", "$B_2$", "$B_3$"]},
    "A2": {"label": "{$diameter\\ of\\ pinholes\\ (mm)$}", "data": [0.2, 1.0, 0.3, 0.6, 0.4]},
    "B1": {"label": "{$d_{min}\\ (mm)$}", "data": [d_min(0.2), d_min(1.0), d_min(0.3), d_min(0.6), d_min(0.4)], "intermed": True},
}

print_standard_table(
    data=data["Ex4"],
    headers=headers,
    column_formats=["l", "2.1", "1.3"],  # CHANGED: first col is text
    caption="Diameters of the pinholes and the calculated minimum diameter for each.",
    label="tab:ex4",
    show=True
)


headers = {
    "beta":  {"label": "{$\\beta$}", "err": 0.1},
    "g_abs": {"label": "{$\\tilde{g}\\ (cm)$}", "err": 0.1},
    "g_rel": {"label": "{$g\\ (cm)$}", "err": 0.2},
    "b":     {"label": "{$b\\ (cm)$}", "err": 0.1},
    "f":     {"label": "{$f\\ (cm)$}", "err": data["Ex1"]["df"]},
}

print_standard_table(
    data=data["Ex1"],
    headers=headers,
    column_formats=["2.1", "2.1", "2.1", "2.1", "2.1"],
    caption="Using Eq.~\\ref{eq:f}, we compute the focal length from measured quantities and compare with the nominal $f$.",
    label="tab:ex1",
    show=False
)


data_blocks_ex2 = [
    {
        "x_abs": data["Ex2"]["x_abs"],
        "x_rel": data["Ex2"]["x_rel"],
        "G": data["Ex2"]["G"],
        "B": data["Ex2"]["B"],
        "beta": data["Ex2"]["beta"],
        "t": data["Ex2"]["t"][0],
        "t/f": data["Ex2"]["t"][0]/4,
        "Gamma_th": data["Ex2"]["Gamma_th"][0],
        "Gamma_exp": data["Ex2"]["Gamma_exp0"],
        "err": {
            "x_abs": 0.1,
            "x_rel": 0.2,
            "G": 0.5,
            "beta": data["Ex2"]["dbeta"],
            "t": 0.1,
            "t/f": 0.1/4,
            "Gamma_exp": data["Ex2"]["dGamma_exp0"],
            "Gamma_th": data["Ex2"]["dGamma_th"],
        },
    },
    {
        "x_abs": data["Ex2"]["x_abs"][1],
        "x_rel": data["Ex2"]["x_rel"][1],
        "G": data["Ex2"]["G2"][1:],
        "B": data["Ex2"]["B2"][1:],
        "beta": data["Ex2"]["beta2"][1:],
        "t": data["Ex2"]["t"][1:],
        "t/f": data["Ex2"]["t"][1:]/4,
        "Gamma_th": data["Ex2"]["Gamma_th"][1:],
        "Gamma_exp": data["Ex2"]["Gamma_exp"][1:],
        "err": {
            "x_abs": 0.1,
            "x_rel": 0.2,
            "G": 0.5,
            "beta": data["Ex2"]["dbeta2"][1:],
            "t": 0.1,
            "t/f": 0.1/4,
            "Gamma_exp": data["Ex2"]["dGamma_exp"][1:],
            "Gamma_th": data["Ex2"]["dGamma_th"],
        },
    },
]


headers = {
    "t":        {"label": "{$t$\\ (cm)}", "repeat": False},
    "x_abs":    {"label": "{$\\tilde{x}$\\ (cm)}", "repeat": True},
    "x_rel":    {"label": "{$x$\\ (cm)}", "repeat": True},
    "G":        {"label": "{$G$}"},
    "B":        {"label": "{$B$}"},
    "beta":     {"label": "{$\\beta_{ob}$}"},
    "t/f":      {"label": "{$t/f$}"},
    "Gamma_th": {"label": "{$\\Gamma_{th}$}", "repeat": True},
    "Gamma_exp":{"label": "{$\\Gamma_{ex}$}"},
}


print_complex_table(
    data=data_blocks_ex2,
    headers=headers,
    column_formats=["2.1"] * len(headers),
    caption=(
        "First, $t$ is fixed at $15\\,\\mathrm{cm}$ while $x'$ is varied to find $x_{\\mathrm{best}}$; "
        "then $x'$ is fixed and $t$ is varied. Magnification is shown from both theory and experiment."
    ),
    label="tab:ex2",
    show=True
)


