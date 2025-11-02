import numpy as np
import sounddevice as sd

import pandas as pd
data = pd.read_csv(r"FP/HOL/Data/S 15.6 Alte Daten/16Uhr52_Interferenz_z=-infcm.dat",
                   header=None, decimal=',').squeeze().to_numpy(dtype=float)
# If comma doesn’t work, try: np.loadtxt(..., delimiter=None) or sep=r'\s+'

import numpy as np
from scipy.io.wavfile import write
from scipy.signal import resample_poly

# data: your 1 kHz samples (mono, shape (N,))
fs_in = 1000

# 1) Remove DC offset (important for audio)
x = data.astype(np.float64)
x = x - np.nanmean(x)

# 2) Normalize safely
peak = np.nanmax(np.abs(x))
if peak > 0:
    x = x / peak

# 3) Upsample to a common audio rate (e.g. 48000 Hz)
fs_out = 48000
y = resample_poly(x, up=fs_out, down=fs_in)

# 4a) Write as 16-bit PCM (widest compatibility)
write("output_48k_int16.wav", fs_out, (y * 0.9 * 32767).astype(np.int16))

# 4b) Or write as float32 WAV (often plays fine, keeps dynamic range)
write("output_48k_float.wav", fs_out, y.astype(np.float32))