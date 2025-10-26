import numpy as np
from scipy.interpolate import UnivariateSpline
from Functions.plotting import plot_data, DatasetSpec

# --- load ---
red_data   = np.loadtxt('red.csv', delimiter=',')
green_data = np.loadtxt('green.csv', delimiter=',')


red_data[:, 0] = np.abs(red_data[:, 0])
green_data[:, 0] = np.abs(green_data[:, 0])

# sort by angle and cast to arrays
def sort_xy(a):
    a = a[np.argsort(a[:,0])]
    return a[:,0].astype(float), a[:,1].astype(float)

red_angle,   red_intensity   = sort_xy(red_data)
green_angle, green_intensity = sort_xy(green_data)



# --- splines (tune s as needed; s=0 → interpolate through points) ---
red_spline   = UnivariateSpline(red_angle,   red_intensity,   s=8, k=3)
green_spline = UnivariateSpline(green_angle, green_intensity, s=8, k=3)

x_red   = np.linspace(red_angle.min(),   red_angle.max(),   361)
x_green = np.linspace(green_angle.min(), green_angle.max(), 361)

s1 = DatasetSpec(x=x_red,   y=red_spline(x_red),   label="Red LED",   line='-', marker="None")
s2 = DatasetSpec(x=x_green, y=green_spline(x_green), label="Green LED", line='-', marker="None")

plot_data(
    datasets=[s1, s2],
    xlabel="Angle (°)",
    ylabel="Relative intensity (%)",
    title="LED angular intensity",
    color_seed=54,
    plot=True
)

# --- IES export (Type-C, rotational symmetry) ---
def save_ies_from_curve(
    filename: str,
    angles_deg: np.ndarray,        # measured angles in deg, either -90..+90 or 0..90
    rel_intensity: np.ndarray,     # relative (0..1 or 0..100)
    step_deg: float = 2.5,         # vertical-angle grid in the IES
    peak_cd: float = 1.0,          # scale to absolute peak candela; keep 1.0 if unknown
    units_meters: bool = True,
    meta: dict | None = None
):
    angles_deg = np.asarray(angles_deg, float)
    vals = np.asarray(rel_intensity, float)
    if vals.max() > 1.5:  # assume percent
        vals = vals / 100.0

    # --- 1) reduce to forward hemisphere 0..90 by mirroring ---
    if angles_deg.min() < 0:
        a = np.abs(angles_deg)
        v = vals
    else:
        a, v = angles_deg, vals
    # keep only 0..90
    keep = (a >= 0) & (a <= 90)
    a, v = a[keep], v[keep]

    # sort + unique
    order = np.argsort(a)
    a, v = a[order], v[order]
    a, idx = np.unique(a, return_index=True)
    v = v[idx]

    # --- 2) resample with a spline on a clean grid 0..90 ---
    grid_fwd = np.arange(0.0, 90.0 + 1e-6, step_deg)
    spl = UnivariateSpline(a, v, s=0, k=min(3, max(1, len(a)-1)))
    fwd = np.clip(spl(grid_fwd), 0.0, None)

    # normalize peak to 1 then scale to peak_cd
    if fwd.max() > 0:
        fwd = fwd / fwd.max() * float(peak_cd)

    # --- 3) append zeros for 90..180 (back hemisphere blocked) ---
    grid_back = np.arange(90.0 + step_deg, 180.0 + 1e-6, step_deg)
    back = np.zeros_like(grid_back)

    vert_angles = np.concatenate([grid_fwd, grid_back])
    candela = np.concatenate([fwd, back])

    # --- 4) write IES (Type C, rotational symmetry) ---
    meta = meta or {}
    with open(filename, "w", encoding="ascii") as f:
        f.write("IESNA:LM-63-2002\n")
        if "TEST" in meta:     f.write(f"[TEST] {meta['TEST']}\n")
        if "MANUFAC" in meta:  f.write(f"[MANUFAC] {meta['MANUFAC']}\n")
        if "LUMCAT" in meta:   f.write(f"[LUMCAT] {meta['LUMCAT']}\n")
        f.write("TILT=NONE\n")

        N_lamps = 1
        lumens_per_lamp = -1            # unknown (like your BEGA example)
        candela_mult = 1.0
        N_vert = len(vert_angles)
        N_horiz = 1                      # rotational symmetry
        photometric_type = 1             # Type C
        units_type = 2 if units_meters else 1
        width = length = height = 0.0
        f.write(f"{N_lamps} {lumens_per_lamp} {candela_mult} {N_vert} {N_horiz} "
                f"{photometric_type} {units_type} {width} {length} {height}\n")

        # ballast factor, future, input watts
        f.write("1.0 1.0 0\n")

        # vertical angles (0..180), then horizontal angle 0, then candela values
        f.write(" ".join(f"{v:.1f}" for v in vert_angles) + "\n")
        f.write("0.0\n")
        f.write(" ".join(f"{c:.4f}" for c in candela) + "\n")

# examples
save_ies_from_curve(
    "red_led.ies",
    angles_deg=x_red,
    rel_intensity=red_spline(x_red),
    step_deg=2.5,
    peak_cd=1.0,                     # put real peak cd here if you have it
    meta={"TEST":"From LED polar curve"}
)


save_ies_from_curve(
    "green_led.ies",
    angles_deg=x_green,
    rel_intensity=green_spline(x_green),
    step_deg=2.5,
    peak_cd=1.0,
    meta={"TEST":"From LED polar curve"}
)
