import numpy as np
import pandas as pd

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from Functions.plotting import *
from Functions.tables import *
from Functions.help import *

source_dir = Path(__file__).resolve().parent / "data"

data = {}
names = []
for f in source_dir.glob("*.csv"):
    t, I = pd.read_csv(source_dir / f, sep=",", header=None).values.T
    data[f.stem] = {"t": t, "I": I}
    names.append(f.stem)


T0 = 273.15 + 19.3
Tf = 273.15 + 19.7

P0 = 101290  # in Pa
Pf = 101330

wavelength = 405e-9  # in m
S = 1 / (wavelength * 1e6) ** 2  # in 1/um


A = 8342.54
B = 2406147
C = 15998
D = 96095.43
F = 0.00972
E = 0.601
G = 0.003661

n_s = 1 + 1e-8 * (A + B / (130 - S) + C / (38.9 - S))


def n_air(T, P):
    t = T - 273.15
    X = (1 + 1e-8 * (E - F * t) * P) / (1 + G * t)
    return 1 + P * (n_s - 1) * X / D


for T, P in zip([T0, Tf], [P0, Pf]):
    n = n_air(T, P)
    print(f"T={T - 273.15:.1f} C and P={P / 1e2:.1f} mbar n = {n:.8f}")

n_avg = (n_air(T0, P0) + n_air(Tf, Pf)) / 2
print(f"Average n = {n_avg:.6f}")

P_avg = (P0 + Pf) / 2
T_avg = (T0 + Tf) / 2
N = P_avg / (1.380649e-23 * T_avg)

beta_th = 24 * np.pi**3 * ((n_avg**2 - 1) / (n_avg**2 + 2)) ** 2 / (wavelength**4 * N)


taus_vac = []
taus_air = []
s_vac = []
s_air = []
tau_scan_specs = []
r2_scan_specs = []

for i, name in enumerate(names):
    t_raw = data[name]["t"]
    I_raw = data[name]["I"]

    t_us = t_raw * 1e6
    start = 100.0
    end = 200.0

    mask = (t_us > start) & (t_us < end)
    t = t_us[mask]
    I = I_raw[mask] * -1e4 + 1.5

    t0 = t.min()

    # Estimate the baseline from the last part of the selected trace.
    c0 = np.mean(I[-max(20, len(I) // 10) :])
    A0 = I[0] - c0

    starts = np.linspace(100.0, 120.0, 200)
    start_scan_vals = []
    tau_scan = []
    tau_err_scan = []
    r2_scan = []

    for start_scan in starts:
        mask_scan = (t_us > start_scan) & (t_us < end)
        t_scan = t_us[mask_scan]
        I_scan = I_raw[mask_scan] * -1e4 + 1.5

        if len(t_scan) < 40:
            continue

        t0_scan = t_scan.min()
        c0_scan = np.mean(I_scan[-max(20, len(I_scan) // 10) :])
        A0_scan = I_scan[0] - c0_scan

        res_scan = lmfit(
            t_scan,
            I_scan,
            model=lambda x, A, c, tau: A * np.exp(-(x - t0_scan) / tau) + c,
            initial_params={"A": A0_scan, "c": c0_scan, "tau": 10.0},
            constraints={
                "tau": {"min": 0},
                "A": {"min": 0},
            },
        )

        start_scan_vals.append(start_scan)
        tau_scan.append(res_scan.params["tau"].value)
        tau_err_scan.append(res_scan.params["tau"].stderr if res_scan.params["tau"].stderr is not None else np.nan)
        r2_scan.append(res_scan.rsquared)


    res = lmfit(
        t,
        I,
        model=lambda x, A, c, tau: A * np.exp(-(x - t0) / tau) + c, 
        initial_params={"A": A0, "c": c0, "tau": 10.0},
        constraints={
            "tau": {"min": 0},
            "A": {"min": 0},
        },
    )

    t_fit = np.linspace(t.min(), t.max(), 1000)
    residuals = I - res.eval(x=t)

    tau = res.params["tau"].value
    dtau = res.params["tau"].stderr
    c_fit = res.params["c"].value

    if "vac" in name:
        taus_vac.append([tau, dtau])
    else:
        taus_air.append([tau, dtau])

    s = DatasetSpec(
        x=t,
        y=I - c_fit,
        marker="None",
        line="-",
        color_group=name,
        label=f"${name}$",
        fit_x=t_fit,
        fit_y=res.eval(x=t_fit) - c_fit,
        fit_color="red",
    )
    if "vac" in name:
        s_vac.append(s)
    else:
        s_air.append(s)

    s_res = DatasetSpec(
        x=t,
        y=residuals,
        marker="None",
        line="-",
        color_group=name,
        label=f"Residuals {name}",
    )

    plot_data(
        s_res,
        height=15,
        width=20,
        color_seed=54,
        title=f"{name}: tau = {tau:.3f} +- {dtau:.3f} us, R^2 = {res.rsquared:.4f}",
        xlabel="Time $(\\mu s)$",
        ylabel="Residuals",
        plot=False,
    )

    tau_scan_specs.append(
        DatasetSpec(
            x=np.array(start_scan_vals),
            y=np.array(tau_scan),
            yerr=np.array(tau_err_scan),
            marker=".",
            line="None",
            color_group=name[:3],
            label=f"${name}$",
        )
    )
    r2_scan_specs.append(
        DatasetSpec(
            x=np.array(start_scan_vals),
            y=np.array(r2_scan),
            marker=".",
            line="None",
            color_group=name[:3],
            label=f"${name}$",
        )
    )


tau_scan_specs_vac = [spec for spec in tau_scan_specs if "vac" in (spec.label or "").lower()]
tau_scan_specs_air = [spec for spec in tau_scan_specs if "vac" not in (spec.label or "").lower()]


def _stats_from_scan_specs(specs_group):
    if len(specs_group) == 0:
        return None

    x_ref = np.asarray(specs_group[0].x, dtype=float)
    y_stack = []
    for spec in specs_group:
        x_i = np.asarray(spec.x, dtype=float)
        y_i = np.asarray(spec.y, dtype=float)
        if x_i.shape != x_ref.shape or not np.allclose(x_i, x_ref):
            continue
        y_stack.append(y_i)

    if len(y_stack) == 0:
        return None

    y_stack = np.asarray(y_stack, dtype=float)
    tau_mean = np.nanmean(y_stack, axis=0)
    tau_std = np.nanstd(y_stack, axis=0)
    return x_ref, tau_mean, tau_std


vac_stats = _stats_from_scan_specs(tau_scan_specs_vac)
air_stats = _stats_from_scan_specs(tau_scan_specs_air)

s_taus_vac = None
if vac_stats is not None:
    x_vac, tau_mean_vac, tau_std_vac = vac_stats
    s_taus_vac = DatasetSpec(
        x=x_vac,
        y=tau_mean_vac,
        confidence=[(tau_mean_vac - tau_std_vac, tau_mean_vac + tau_std_vac)],
        confidence_label=False,
        marker="None",
        line="-",
        color="black",
        color_group="average_vac",
        label="Avg (vac)",
    )
    i_vac = int(np.nanargmin(tau_std_vac))
    print(f"Min std (vac) at start = {x_vac[i_vac]:.3f} us: sigma = {tau_std_vac[i_vac]:.4f} us")

s_taus_air = None
if air_stats is not None:
    x_air, tau_mean_air, tau_std_air = air_stats
    s_taus_air = DatasetSpec(
        x=x_air,
        y=tau_mean_air,
        confidence=[(tau_mean_air - tau_std_air, tau_mean_air + tau_std_air)],
        confidence_label=False,
        marker="None",
        line="-",
        color="gray",
        color_group="average_air",
        label="Avg (air)",
    )
    i_air = int(np.nanargmin(tau_std_air))
    print(f"Min std (air) at start = {x_air[i_air]:.3f} us: sigma = {tau_std_air[i_air]:.4f} us")


std_specs = []
if vac_stats is not None:
    std_specs.append(
        DatasetSpec(
            x=x_vac,
            y=tau_std_vac,
            marker=".",
            line="-",
            color="black",
            color_group="std_vac",
            label="Std (vac)",
        )
    )
if air_stats is not None:
    std_specs.append(
        DatasetSpec(
            x=x_air,
            y=tau_std_air,
            marker=".",
            line="-",
            color="gray",
            color_group="std_air",
            label="Std (air)",
        )
    )

if std_specs:
    plot_data(
        std_specs,
        height=9,
        width=12,
        color_seed=52,
        xlabel="Fit start $(\\mu s)$",
        ylabel="$\\sigma_\\tau \\, (\\mu s)$",
        title="Std of tau vs fit start",
        filename="Plots/tau_std_vs_start.pdf",
        plot=False,
    )




plot_data(
    tau_scan_specs + [s for s in (s_taus_vac, s_taus_air) if s is not None],
    height=15,
    width=20,
    color_seed=52,
    xlabel="Fit start $(\\mu s)$",
    ylabel="$\\tau \\, (\\mu s)$",
    title="Scan of fit start time for tau (vac & air averages)",
    filename="Plots/tau_vs_start.pdf",
    plot=False,
)



plot_data(
    s_vac,
    height=15,
    width=20,
    color_seed=54,
    xlabel="Time $(\\mu s)$",
    ylabel="Intensity ",
    filename="Plots/vac.pdf",
    plot=False,
)

plot_data(
    s_air,
    height=15,
    width=20,
    color_seed=54,
    xlabel="Time $(\\mu s)$",
    ylabel="Intensity ",
    filename="Plots/air.pdf",
    plot=False,
)

s2_vac = DatasetSpec(
    x=np.arange(len(taus_vac)) + 1,
    y=np.array(taus_vac)[:, 0],
    yerr=np.array(taus_vac)[:, 1],
    marker=".",
    line="None",
    color_group="vacuum",
)
s2_air = DatasetSpec(
    x=np.arange(len(taus_air)),
    y=np.array(taus_air)[:, 0],
    yerr=np.array(taus_air)[:, 1],
    marker=".",
    line="None",
    color_group="air",
)

tau_vac = np.mean(np.array(taus_vac)[:, 0]), np.std(np.array(taus_vac)[:, 0])
tau_air = np.mean(np.array(taus_air)[:, 0]), np.std(np.array(taus_air)[:, 0])
print(f"Average tau in vacuum = {tau_vac[0]:.3f} +- {tau_vac[1]:.3f} us")
print(f"Average tau in air = {tau_air[0]:.3f} +- {tau_air[1]:.3f} us")

c = 299792458
beta_exp = 1 / c * (1 / tau_air[0] - 1 / tau_vac[0]) * 1e6
beta_exp_err = 1 / c * np.sqrt((tau_air[1] / tau_air[0] ** 2) ** 2 + (tau_vac[1] / tau_vac[0] ** 2) ** 2) * 1e6

print(f"Theoretical beta = {beta_th * 1e5:.2f} 10^5 1/m")
print(f"Experimental beta = {beta_exp * 1e5:.2f} +- {beta_exp_err * 1e5:.2f} 10^5 1/m")

plot_data(
    [s2_vac, s2_air],
    height=15,
    width=20,
    color_seed=50,
    ylabel="$\\tau \\, (\\mu s)$",
    filename="Plots/tau.pdf",
    plot=False,
)

plot_data(
    r2_scan_specs,
    height=15,
    width=20,
    color_seed=52,
    xlabel="Fit start $(\\mu s)$",
    ylabel="$R^2$",
    filename="Plots/r2_vs_start.pdf",
    plot=False,
)
