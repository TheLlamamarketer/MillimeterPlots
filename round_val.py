import numpy as np


def FirstSignificant(x):
    if x == 0:
        return 0
    return -int(np.floor(np.log10(abs(x))))


def round_val(val, err=0, intermed=True):
    if val == 0:
        return 0, err

    power = FirstSignificant(err if err else val)
    n = 2 if intermed else 1

    if err == 0:
        err_round = 0
        power = power + (n - 1) if power > 0 else n

    else:
        power += n - 1
        factor = 10**power
        err_round = np.ceil(err * factor) / factor

    return round(val, power), err_round
