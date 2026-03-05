import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.table import Table, vstack, unique
from astroquery.gaia import Gaia
from matplotlib.ticker import FuncFormatter

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from Functions.wavelength_colors import temperature_to_rgb


# ----------------------------- user settings -----------------------------
CACHE_PATH = "gaia_hr_sample.fits"
FORCE_REFETCH = False

RANDOM_SEED = 1

# Clean sample for crisp main sequence and a good white dwarf locus
N_CHUNKS_CLEAN = 50
TOP_PER_CHUNK_CLEAN = 3000
CHUNK_SPAN = 50_000_000

# Optional second layer to emphasize giants
INCLUDE_GIANTS_LAYER = True
N_CHUNKS_GIANTS = 50
TOP_PER_CHUNK_GIANTS = 2500

# Plot tuning
POINT_SIZE_CLEAN = 2
ALPHA_CLEAN = 0.22

POINT_SIZE_GIANTS = 2.5
ALPHA_GIANTS = 0.12

# Temperature to RGB lookup table
T_LUT_MIN = 2500.0
T_LUT_MAX = 50000.0
T_LUT_N = 1500


# ----------------------------- helpers -----------------------------
def to_array(col):
    if hasattr(col, "filled"):
        return np.asarray(col.filled(np.nan), dtype=float)
    return np.asarray(col, dtype=float)


def fetch_chunk(r0, r1, top, where_extra=""):
    where_extra = where_extra.strip()
    if where_extra:
        where_extra = " AND " + where_extra

    query = f"""
    SELECT TOP {top}
        source_id, random_index,
        bp_rp, phot_g_mean_mag, parallax,
        teff_gspphot,
        ruwe, parallax_over_error
    FROM gaiadr3.gaia_source_lite
    WHERE random_index BETWEEN {int(r0)} AND {int(r1)}
      AND bp_rp IS NOT NULL
      AND phot_g_mean_mag IS NOT NULL
      AND parallax IS NOT NULL
      AND parallax > 0
      AND ruwe < 1.4
      AND phot_g_mean_mag < 20
      {where_extra}
    """
    return Gaia.launch_job(query).get_results()


def fetch_random_sample(n_chunks, top_per_chunk, chunk_span, where_extra=""):
    rng = np.random.default_rng(RANDOM_SEED)
    chunks = []
    for _ in range(n_chunks):
        a = int(rng.integers(0, 2_000_000_000 - chunk_span))
        b = a + int(chunk_span)
        chunks.append(fetch_chunk(a, b, top=top_per_chunk, where_extra=where_extra))
    t = vstack(chunks, metadata_conflicts="silent")
    if "source_id" in t.colnames:
        t = unique(t, keys="source_id")
    return t


def compute_absolute_magnitude(G, plx_mas):
    plx = plx_mas * u.mas
    d_pc = plx.to(u.pc, equivalencies=u.parallax()).value
    M_G = G - 5.0 * np.log10(d_pc) + 5.0
    return M_G


def build_bp_teff_mapping(bp_rp, teff, M_G):
    """
    Empirical mapping from BP-RP to Teff using a main sequence like locus.
    This is used for the top axis and for filling missing Teff values.
    """
    has_teff = np.isfinite(teff) & (teff > 0)

    sel = (
        has_teff
        & np.isfinite(bp_rp)
        & np.isfinite(M_G)
        & (bp_rp > -0.5)
        & (bp_rp < 5.5)
        & (teff > T_LUT_MIN)
        & (teff < T_LUT_MAX)
        & (M_G > 2.0)
        & (M_G < 10.5)
    )

    bp_map = bp_rp[sel]
    T_map = teff[sel]

    if bp_map.size < 200:
        return None

    bins = np.linspace(-0.5, 5.5, 220)
    centers = 0.5 * (bins[:-1] + bins[1:])
    T_med = np.full_like(centers, np.nan, dtype=float)

    for i in range(centers.size):
        m = (bp_map >= bins[i]) & (bp_map < bins[i + 1])
        if np.any(m):
            T_med[i] = np.nanmedian(T_map[m])

    good = np.isfinite(T_med)
    centers = centers[good]
    T_med = T_med[good]

    if centers.size < 2:
        return None

    # Enforce monotonic decrease of Teff with increasing BP-RP
    T_mono = np.minimum.accumulate(T_med)

    return centers, T_mono


def make_temperature_lut():
    T_lut = np.linspace(T_LUT_MIN, T_LUT_MAX, T_LUT_N)
    rgb = np.array([temperature_to_rgb(T) for T in T_lut], dtype=float)
    if np.nanmax(rgb) > 1.0:
        rgb /= 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return T_lut, rgb


def apply_temperature_to_rgb(T, T_lut, rgb_lut):
    T_clip = np.clip(T, T_lut.min(), T_lut.max())
    colors = np.empty((T_clip.size, 3), dtype=float)
    for k in range(3):
        colors[:, k] = np.interp(T_clip, T_lut, rgb_lut[:, k])
    return np.clip(colors, 0.0, 1.0)


# ----------------------------- main -----------------------------
def main():
    if (not FORCE_REFETCH) and Path(CACHE_PATH).exists():
        t_all = Table.read(CACHE_PATH)
        print(f"loaded cache: {len(t_all)}")
    else:
        print("fetching clean sample")
        t_clean = fetch_random_sample(
            n_chunks=N_CHUNKS_CLEAN,
            top_per_chunk=TOP_PER_CHUNK_CLEAN,
            chunk_span=CHUNK_SPAN,
            where_extra="parallax_over_error > 5",
        )
        t_clean["layer"] = np.full(len(t_clean), "clean", dtype="U8")

        tables = [t_clean]

        if INCLUDE_GIANTS_LAYER:
            print("fetching giants emphasis sample")
            # Mildly looser SNR, plus an absolute magnitude cut to concentrate giants
            # M_G = G + 5*log10(parallax_mas) - 10
            t_giants = fetch_random_sample(
                n_chunks=N_CHUNKS_GIANTS,
                top_per_chunk=TOP_PER_CHUNK_GIANTS,
                chunk_span=CHUNK_SPAN,
                where_extra="parallax_over_error > 1 AND (phot_g_mean_mag + 5*log10(parallax) - 10) < 3",
            )
            t_giants["layer"] = np.full(len(t_giants), "giants", dtype="U8")
            tables.append(t_giants)

        t_all = vstack(tables, metadata_conflicts="silent")
        if "source_id" in t_all.colnames:
            t_all = unique(t_all, keys="source_id")

        print(f"fetched total: {len(t_all)}")
        t_all.write(CACHE_PATH, overwrite=True)

    bp_rp = to_array(t_all["bp_rp"])
    G = to_array(t_all["phot_g_mean_mag"])
    plx_mas = to_array(t_all["parallax"])
    teff = to_array(t_all["teff_gspphot"])
    layer = np.asarray(t_all["layer"]).astype(str) if "layer" in t_all.colnames else np.full(bp_rp.size, "all")

    M_G = compute_absolute_magnitude(G, plx_mas)

    finite = np.isfinite(bp_rp) & np.isfinite(M_G)
    bp_rp = bp_rp[finite]
    M_G = M_G[finite]
    teff = teff[finite]
    layer = layer[finite]

    print("after finite mask:", bp_rp.size)

    if bp_rp.size == 0:
        raise RuntimeError("No valid data after filtering")

    # Build mapping for top axis and for Teff fill
    mapping = build_bp_teff_mapping(bp_rp, teff, M_G)

    if mapping is None:
        centers = None
        T_mono = None
        print("warning: not enough valid temperature data for a stable BP-RP to Teff mapping")
    else:
        centers, T_mono = mapping

    def bp_to_teff(x):
        if centers is None or T_mono is None:
            return np.full_like(x, 5000.0, dtype=float)
        return np.interp(x, centers, T_mono, left=T_mono[0], right=T_mono[-1])

    def teff_to_bp(x):
        if centers is None or T_mono is None:
            return np.full_like(x, 1.0, dtype=float)
        return np.interp(x, T_mono[::-1], centers[::-1], left=centers[-1], right=centers[0])

    # Choose Teff used for color
    has_teff = np.isfinite(teff) & (teff > 0)

    T_color = teff.copy()
    missing = ~has_teff

    # Fill missing Teff from empirical mapping
    if centers is not None and T_mono is not None:
        T_color[missing] = bp_to_teff(bp_rp[missing])
    else:
        T_color[missing] = 5000.0

    # White dwarf heuristic for missing Teff, improves the blue faint sequence appearance
    wd_like = missing & (M_G > 9.0) & (bp_rp < 1.0)
    if np.any(wd_like):
        T_wd = 20000.0 - 12000.0 * np.clip(bp_rp[wd_like], 0.0, 1.0)
        T_color[wd_like] = np.clip(T_wd, 6000.0, T_LUT_MAX)

    # Final clip
    T_color = np.clip(T_color, T_LUT_MIN, T_LUT_MAX)

    # Colors from LUT
    T_lut, rgb_lut = make_temperature_lut()
    colors = apply_temperature_to_rgb(T_color, T_lut, rgb_lut)

    # Plot
    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    idx_giants = layer == "giants"
    idx_clean = layer == "clean"
    idx_other = ~(idx_giants | idx_clean)

    if np.any(idx_giants):
        ax.scatter(
            bp_rp[idx_giants],
            M_G[idx_giants],
            s=POINT_SIZE_GIANTS,
            alpha=ALPHA_GIANTS,
            c=colors[idx_giants],
            linewidths=0,
            rasterized=True,
        )

    if np.any(idx_clean):
        ax.scatter(
            bp_rp[idx_clean],
            M_G[idx_clean],
            s=POINT_SIZE_CLEAN,
            alpha=ALPHA_CLEAN,
            c=colors[idx_clean],
            linewidths=0,
            rasterized=True,
        )

    if np.any(idx_other):
        ax.scatter(
            bp_rp[idx_other],
            M_G[idx_other],
            s=POINT_SIZE_CLEAN,
            alpha=ALPHA_CLEAN,
            c=colors[idx_other],
            linewidths=0,
            rasterized=True,
        )

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(16, -6)

    ax.set_xlabel("Gaia colour BP - RP (mag)", color="white")
    ax.set_ylabel("Absolute magnitude $M_G$", color="white")
    ax.set_title("Gaia DR3 HR diagram", color="white")
    ax.tick_params(colors="white")
    ax.grid(alpha=0.25, color="white")

    if centers is not None and T_mono is not None and T_mono.size >= 2:
        secax = ax.secondary_xaxis("top", functions=(bp_to_teff, teff_to_bp))
        secax.set_xlabel("Effective Temperature (K)", color="white")
        secax.tick_params(colors="white")

        T_ticks = np.array([50000, 20000, 10000, 7500, 6000, 5000, 4000, 3500, 3000, 2500], dtype=float)
        Tmin, Tmax = float(np.nanmin(T_mono)), float(np.nanmax(T_mono))
        T_ticks = T_ticks[(T_ticks >= Tmin) & (T_ticks <= Tmax)]
        if T_ticks.size > 0:
            secax.set_xticks(T_ticks)
            secax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):d}"))

    plt.tight_layout()
    plt.savefig("gaia_hr_diagram.png", dpi=300)
    plt.show(block=True)
   


if __name__ == "__main__":
    main()
