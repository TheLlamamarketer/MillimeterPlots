import numpy as np
from PIL import Image
from scipy.special import eval_hermite

# ---------------- user parameters ----------------
size = 320          # output image: size x size pixels
I_0 = 1.0           # intensity scale
w = size / 4.0      # beam waist scale in pixels

m = 0               # Hermite index in x
n = 1               # Hermite index in y

# Optional: rotate the pattern (degrees). Keep 0.0 for no rotation.
theta_deg = 0.0

# Optional: different waists (astigmatism). Keep wy = wx for symmetric.
wx = w
wy = w

modes = [(0,0), (0,1), (1,0), (1,1)]

for m, n in modes:
    out_name = f"TEM_m{m}_n{n}.png"

    # ---------------- helpers ----------------
    def linear_to_sRGB(lin: float) -> int:
        # Formula from http://www.w3.org/Graphics/Color/sRGB
        if lin <= 0.00304:
            lin = 12.92 * lin
        else:
            lin = 1.055 * (lin ** (1.0 / 2.4)) - 0.055
        lin = 0.0 if lin < 0.0 else (1.0 if lin > 1.0 else lin)
        return int(round(255.0 * lin))

    # ---------------- generate field ----------------
    # Pixel coordinates centered at (0,0)
    xs = np.arange(size) - (size - 1) / 2.0
    ys = np.arange(size) - (size - 1) / 2.0
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    # Rotation
    theta = np.deg2rad(theta_deg)
    ct, st = np.cos(theta), np.sin(theta)
    Xr = ct * X + st * Y
    Yr = -st * X + ct * Y

    # Hermite Gaussian field amplitude (up to an overall constant)
    # u_mn(x,y) ∝ H_m( sqrt(2) x / wx ) H_n( sqrt(2) y / wy ) exp( -x^2/wx^2 - y^2/wy^2 )
    Hx = eval_hermite(m, np.sqrt(2.0) * Xr / wx)
    Hy = eval_hermite(n, np.sqrt(2.0) * Yr / wy)
    U = Hx * Hy * np.exp(-(Xr**2) / (wx**2) - (Yr**2) / (wy**2))

    # Intensity
    I = I_0 * (U**2)

    # Normalize
    Imax = float(I.max()) if float(I.max()) > 0.0 else 1.0
    In = I / Imax
    
    bg = 255

    # Convert to purple-tinted RGB
    cooked = Image.new("RGB", (size, size))
    pix = cooked.load()
    for j in range(size):
        for i in range(size):
            v = float(In[j, i])          
            s = linear_to_sRGB(v)        

            pr = s
            pg = int(0 * s)
            pb = int(0 * s)

            # alpha blend: out = (1-a)*bg + a*purple
            r = int((1.0 - v) * bg + v * pr)
            g = int((1.0 - v) * bg + v * pg)
            b = int((1.0 - v) * bg + v * pb)

            pix[i, j] = (r, g, b)

    cooked.save(out_name)
    print("saved " + out_name)
