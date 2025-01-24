import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import Model
from decimal import Decimal, getcontext
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

def round_val(val, err=0, intermed=True):
    if val == 0:
        return 0, err

    power = FirstSignificant(err if err else val)
    n = 2 if intermed else 1

    if err == 0:
        power = power + (n - 1) if power > 0 else n
        return round(val, power), power

    else:
        power += n - 1
        factor = 10**power
        err_round = np.ceil(err * factor) / factor
        power = FirstSignificant(err_round)
        if intermed: power += 1
        return round(val, power), err_round, power

def support(xdata, ydata, yerr=None):
    if yerr is None:
        # Unweighted sums
        n = len(xdata)
        Sx = np.sum(xdata)
        Sy = np.sum(ydata)
        Sxx = np.sum(xdata * xdata)
        Sxy = np.sum(xdata * ydata)
        Sw = n  
    else:
        if np.any(yerr == 0):
            raise ValueError("Zero error encountered in yerr. Cannot perform weighted regression with zero errors.")
        weights = 1.0 / (yerr ** 2)
        Sx = np.sum(weights * xdata)
        Sy = np.sum(weights * ydata)
        Sxx = np.sum(weights * xdata * xdata)
        Sxy = np.sum(weights * xdata * ydata)
        Sw = np.sum(weights)
    return Sx, Sxx, Sy, Sxy, Sw

def slope(xdata, ydata, yerr=None):
    Sx, Sxx, Sy, Sxy, Sw = support(xdata, ydata, yerr)
    denominator = Sw * Sxx - Sx ** 2
    if denominator == 0:
        raise ValueError("Denominator is zero; cannot compute slope and intercept.")

    b = (Sw * Sxy - Sx * Sy) / denominator
    a = (Sxx * Sy - Sx * Sxy) / denominator

    # Calculate errors in slope and intercept
    if yerr is None:
        # Unweighted case
        n = len(xdata)
        residuals = ydata - (a + b * xdata)
        variance = np.sum(residuals ** 2) / (n - 2)
        sb = np.sqrt(variance * Sw / denominator)
        sa = np.sqrt(variance * Sxx / denominator)
    else:
        # Weighted case
        weights = 1.0 / (yerr ** 2)
        residuals = ydata - (a + b * xdata)
        variance = np.sum(weights * residuals ** 2) / (Sw - 2)
        sb = np.sqrt(variance * Sw / denominator)
        sa = np.sqrt(variance * Sxx / denominator)

    # Coefficient of determination R^2
    if yerr is None:
        y_mean = np.mean(ydata)
        total_variance = np.sum((ydata - y_mean) ** 2)
        explained_variance = np.sum((a + b * xdata - y_mean) ** 2)
    else:
        y_mean = np.sum(weights * ydata) / Sw
        total_variance = np.sum(weights * (ydata - y_mean) ** 2)
        explained_variance = np.sum(weights * (a + b * xdata - y_mean) ** 2)

    if total_variance == 0:
        R2 = 1.0
    else:
        R2 = explained_variance / total_variance


    return a, sa, b, sb, R2, variance

def linear_fit(xdata, ydata, yerr=None, model="linear", constraints=None):
    """
    Parameters:
    - model: "linear" or "quadratic" (default is "linear").
    - constraints: Dictionary of parameter constraints, e.g., {"a": 0}.
    """
    # Ensure yerr is iterable
    if yerr is not None and not hasattr(yerr, '__len__'):
        yerr = np.array([yerr])

    # Remove NaN and infinite values
    mask = ~np.isnan(xdata) & ~np.isnan(ydata) & np.isfinite(xdata) & np.isfinite(ydata)
    if yerr is not None:
        mask &= ~np.isnan(yerr) & np.isfinite(yerr)
    xdata = xdata[mask]
    ydata = ydata[mask]
    if yerr is not None:
        yerr = yerr[mask]

    if model == "linear":
        def model_func(x, a, b): return a + b * x
    elif model == "quadratic":
        def model_func(x, a, b, c): return a + b * x + c * x ** 2
    else:
        raise ValueError(f"Unsupported model: {model}")
    model = Model(model_func)
    
    params = model.make_params(a=0, b=1, c=1)
    if constraints:
        for param, value in constraints.items():
            if value is not None:
                params[param].set(value=value, vary=False) 

    if yerr is not None:
        if np.any(np.isnan(yerr)) or np.any(np.isnan(xdata)) or np.any(np.isnan(ydata)):
            raise ValueError("NaN values detected in input data.")
        result = model.fit(ydata, params, x=xdata, weights=1.0 / yerr**2)
    else:
        if np.any(np.isnan(xdata)) or np.any(np.isnan(ydata)):
            raise ValueError("NaN values detected in input data.")
        result = model.fit(ydata, params, x=xdata)

    return result

def calc_CI(result, xdata, sigmas=[1, 2]):
    """
    Calculate confidence intervals for the fit using eval_uncertainty.
    Returns:
    - ci_list: List of tuples with (lower, upper) bounds for the fit curve at different sigma levels.
    """
    ci_list = []
    try:
        for sigma in sigmas:
            # Calculate uncertainty at the given sigma level
            uncertainty = result.eval_uncertainty(sigma=sigma, x=xdata)
            best_fit = result.eval(x=xdata)

            lower = best_fit - uncertainty
            upper = best_fit + uncertainty

            ci_list.append((lower, upper))
    except Exception as e:
        import logging
        logging.error(f"Error calculating confidence intervals: {e}")
        for sigma in sigmas:
            ci_list.append((np.full_like(xdata, np.nan), np.full_like(xdata, np.nan)))
    return ci_list

def extract_params(result):
    params_dict = {}
    for param in result.params:
        value = result.params[param].value
        stderr = result.params[param].stderr
        params_dict[param] = (value, stderr)
    return params_dict

def calc_R2(result):
    return result.rsquared

def print_result(a, b, da, db, R2, s2):
    print(f"Steigung {{b}} = {b} \u00B1 {db}")
    if a != 0 and da != 0:
        print(f"Achsenabschnitt {{a}} = {a} \u00B1 {da}")
    print(f"Bestimmtheitsmaß {{R^2}} = {R2}")
    print(f"Varianz {{s^2}} = {s2}")
    print()


def plot(x, y, dy, a, b, da, db, xlabel="x", ylabel="y", title=None):
    plt.errorbar(x, y, yerr=dy, fmt="kx", capsize=6, capthick=1, label="Datenpunkte")  
    if da != 0 and db != 0:
        plt.plot(x, a + da + (b - db) * x, "r--", label="Grenzgeraden")
        plt.plot(x, a - da + (b + db) * x, "r--")  
    if a != 0 and da != 0:
        plt.plot(x, a + b * x, "g-", label="Ausgleichsgerade")  
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.legend(loc="best")
    plt.show()
