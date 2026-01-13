import numpy as np
import pandas as pd
from lmfit import Model
from decimal import Decimal, getcontext
import logging 
getcontext().prec = 50

def FirstSignificant(x):
    if x == 0:
        return 0
    return -int(np.floor(np.log10(abs(x))))

def LastSignificant(x):
    if x == 0:
        return 0
    d = Decimal(str(x))
    fractional_part = str(d).split('.')[-1] if '.' in str(d) else ''
    return len(fractional_part)

def last_digit(num):
    if isinstance(num, np.ndarray):
        return np.vectorize(last_digit)(num)
    elif isinstance(num, (int, float, Decimal)):
        if isinstance(num, (float, Decimal)):
            return 10**-LastSignificant(num)
        else: 
            return 1
    else:
        raise TypeError("Input must be an int, float, Decimal or np.ndarray")


def round_val(val: float | int , err: float | int | None = 0, intermed: bool = True):
    if val == 0:
        # Always return (rounded_value, rounded_error, power)
        return 0, err, 0

    power = FirstSignificant(err if err else val)
    n = 2 if intermed else 1

    if err == 0:
        power = power + (n - 1) if power > 0 else n
        return round(val, power), 0, power

    else:
        power += n - 1
        factor = 10**power
        err_round = np.ceil(err * factor) / factor
        power = FirstSignificant(err_round)
        if intermed: power += 1
        return round(val, power), err_round, power
    
def print_round_val(val, err=0, intermed=True):
    if err == 0:
        rounded_val, _, _ = round_val(val, err=0, intermed=intermed)
        return rounded_val
    else:
        rounded_val, rounded_err, power = round_val(val, err=err, intermed=intermed)
        return f"{rounded_val:.{max(0, power)}f} \\pm {rounded_err:.{max(0, power)}f}"



def slope(xdata, ydata, yerr=None):
    if yerr is None: yerr = np.ones_like(ydata)
    mask = ~np.isnan(xdata) & ~np.isnan(ydata) & ~np.isnan(yerr) & np.isfinite(xdata) & np.isfinite(ydata) & np.isfinite(yerr)
    yerr = yerr[mask]
    xdata = xdata[mask]
    ydata = ydata[mask]

    weights = np.where(yerr == 0, 0, 1.0 / (yerr ** 2))

    Sx = np.sum(weights * xdata)
    Sy = np.sum(weights * ydata)
    Sxx = np.sum(weights * xdata * xdata)
    Sxy = np.sum(weights * xdata * ydata)
    Sw = np.sum(weights)

    denominator = Sw * Sxx - Sx ** 2
    if denominator == 0:
        raise ValueError("Denominator is zero; cannot compute slope and intercept.")

    b = (Sw * Sxy - Sx * Sy) / denominator
    a = (Sxx * Sy - Sx * Sxy) / denominator

    n = Sw**2 / np.sum(weights**2)

    residuals = ydata - (a + b * xdata)
    variance = np.sum(weights * residuals ** 2) / (n - 2)
    db = np.sqrt(variance * Sw / denominator)
    da = np.sqrt(variance * Sxx / denominator)

    y_mean = np.sum(weights * ydata) / Sw
    total_variance = np.sum(weights * (ydata - y_mean) ** 2)
    explained_variance = np.sum(weights * (a + b * xdata - y_mean) ** 2)

    if total_variance == 0:
        R2 = 1.0
    else:
        R2 = explained_variance / total_variance

    return a, da, b, db, R2, variance

def lmfit(xdata, ydata, yerr = None, model: str | callable = "linear", constraints = None, initial_params=None):
    """
    Fit data to a specified model: linear, quadratic, or exponential.
    - constraints: Dictionary of parameter constraints, e.g., {"a": 0}.
    """

    x = np.asarray(xdata, dtype=float)
    y = np.asarray(ydata, dtype=float)

    if x.shape != y.shape:
        raise ValueError("xdata and ydata must have the same shape.")
    
    if yerr is None:
        yerr_arr = None
    else:
        yerr_arr = np.asarray(yerr, dtype=float)
        if yerr_arr.shape != y.shape:
            if yerr_arr.size == 1:
                yerr_arr = np.full_like(y, yerr_arr.item())
            else:
                raise ValueError("yerr must have the same shape as ydata or be a single value.")
    
    mask = ~np.isnan(x) & ~np.isnan(y) & np.isfinite(x) & np.isfinite(y)
    if yerr_arr is not None:
        mask &= ~np.isnan(yerr_arr) & np.isfinite(yerr_arr)
    x, y = x[mask], y[mask]
    if yerr_arr is not None:
        yerr_arr = yerr_arr[mask]

    models = {
        "linear": (lambda x, a, b: a + b * x, {"a": 0, "b": 1}),
        "quadratic": (lambda x, a, b, c: a + b * x + c * x**2, {"a": 0, "b": 1, "c": 1}),
        "exponential": (lambda x, a, b, c: a * np.exp(b * x) + c, {"a": np.max(y), "b": 0.2, "c": np.min(y)}),
        "exponential_decay": (lambda x, a, b, c: a * (1 - np.exp(-b * x + c)), {"a": np.max(y), "b": 0.7, "c": 0}),
        "gaussian": (lambda x, a, b, c, d: a * np.exp(-((x - b) / c)**2 / 2) + d, {"a": np.max(y), "b": np.mean(x), "c": 1, "d": np.min(y)})
    }

    if isinstance(model, str):
        if model in models:
            model_func, init = models[model]
            if initial_params:
                init.update(initial_params)
        else:
            raise ValueError(f"Model '{model}' is not recognized. Available models: {list(models.keys())}")
    else:
        model_func = model
        init = initial_params if initial_params is not None else {}
        
    mode_func = Model(model_func)
    params = mode_func.make_params(**init)

    if constraints:
        for param, value in constraints.items():
            if value is None:
                continue
            if isinstance(value, dict): 
                params[param].set(**value)
            else:
                params[param].set(value=value, vary=False)
    
    if yerr_arr is None:
        return mode_func.fit(y, params, x=x)
    
    sigma_floor = 1e-12
    yerr_arr = np.where(yerr_arr < sigma_floor, sigma_floor, yerr_arr)
    weights = 1.0 / yerr_arr

    return mode_func.fit(y, params, x=x, weights=weights)

def calc_CI(result, xdata, sigmas=(1,)):
    """
    Returns {sigma: (lower, upper)} for each requested sigma.
    """
    x = np.asarray(xdata, dtype=float)
    out = {}
    best = result.eval(x=x)

    for s in sigmas:
        if s < 0:
            raise ValueError("Sigma levels must be positive integers.")
        uncertainty = result.eval_uncertainty(sigma=s, x=x)
        out[s] = (best - uncertainty, best + uncertainty)

    return out

def extract_params(result):
    params_dict = {}
    for param in result.params:
        value = result.params[param].value
        stderr = result.params[param].stderr
        params_dict[param] = (value, stderr)
    return params_dict

def calc_R2(result):
    return result.rsquared

def fit_summary_string(result) -> str:
    lines = ["Parameters and their errors:"]
    for name, p in result.params.items():
        val = p.value
        err = p.stderr
        if err is None or not np.isfinite(err):
            rel = "n/a"
            err_str = "n/a"
        else:
            err_str = f"{err:8.4g}"
            rel = "inf" if val == 0 else f"{abs(err/val):.2%}"
        lines.append(f"{name:>8}: {val:6.4g} ± {err_str}  ({rel})")
    try:
        lines.append(f"\nR^2: {result.rsquared}")
    except Exception:
        pass
    return "\n".join(lines)
