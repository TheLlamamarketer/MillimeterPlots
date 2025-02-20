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
    
def print_round_val(val, err=0, intermed=True):
    if err == 0:
        rounded_val, _ = round_val(val, intermed=intermed)
        return rounded_val
    else:
        rounded_val, rounded_err, _ = round_val(val, err, intermed=intermed)
        return f"{rounded_val} \\pm {rounded_err}"

def calc_sums(xdata, ydata, weights):
    Sx = np.sum(weights * xdata)
    Sy = np.sum(weights * ydata)
    Sxx = np.sum(weights * xdata * xdata)
    Sxy = np.sum(weights * xdata * ydata)
    Sw = np.sum(weights)
    return Sx, Sxx, Sy, Sxy, Sw

def slope(xdata, ydata, yerr=None):
    if yerr is None: yerr = np.ones_like(ydata)
    mask = ~np.isnan(xdata) & ~np.isnan(ydata) & ~np.isnan(yerr) & np.isfinite(xdata) & np.isfinite(ydata) & np.isfinite(yerr)
    yerr = yerr[mask]
    xdata = xdata[mask]
    ydata = ydata[mask]

    weights = np.where(yerr == 0, 0, 1.0 / (yerr ** 2))

    Sx, Sxx, Sy, Sxy, Sw = calc_sums(xdata, ydata, weights)
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

def lmfit(xdata, ydata, yerr=None, model="linear", constraints=None, const_weight=1):
    """
    Fit data to a specified model: linear, quadratic, or exponential.
    - constraints: Dictionary of parameter constraints, e.g., {"a": 0}.
    """
    if yerr is not None and not hasattr(yerr, "__len__"):
        yerr = np.array([yerr])

    # Remove NaN and infinite values
    mask = ~np.isnan(xdata) & ~np.isnan(ydata) & np.isfinite(xdata) & np.isfinite(ydata)
    if yerr is not None:
        mask &= ~np.isnan(yerr) & np.isfinite(yerr)
        yerr = yerr[mask]
    xdata, ydata = xdata[mask], ydata[mask]

    # Define models and initial parameters
    models = {
        "linear": (lambda x, a, b: a + b * x, {"a": 0, "b": 1}),
        "quadratic": (lambda x, a, b, c: a + b * x + c * x**2, {"a": 0, "b": 1, "c": 1}),
        "exponential": (lambda x, a, b, c: a * np.exp(b * x) + c, {"a": np.max(ydata), "b": 0.2, "c": np.min(ydata)}),
        "exponential_decay": (lambda x, a, b, c: a * (1 - np.exp(-b * x + c)), {"a": np.max(ydata), "b": 0.7, "c": 0}),
        "gaussian": (lambda x, a, b, c, d: a * np.exp(-((x - b) / c)**2 / 2) + d, {"a": np.max(ydata), "b": np.mean(xdata), "c": 1, "d": np.min(ydata)})
    }

    if model not in models:
        raise ValueError(f"Unrecognized model: {model}")

    model_func, init_params = models[model]
    mode_func = Model(model_func)
    params = mode_func.make_params(**init_params)

    if constraints:
        for param, value in constraints.items():
            if value is not None:
                if isinstance(value, dict):  # Allow setting min/max
                    params[param].set(**value)
                else:
                    params[param].set(value=value, vary=False)


    if yerr is not None:
        weights = np.where(yerr == 0, 0, 1.0 / yerr**2)
        w = sum(weights)/len(weights)
        result = mode_func.fit(ydata, params, x=xdata, weights=weights + const_weight/w)
    else:
        result = mode_func.fit(ydata, params, x=xdata)

    #print(result.fit_report())

    return result

def calc_CI(result, xdata, sigmas=[1]):
    """
    Calculate confidence intervals for the fit using eval_uncertainty.
    Returns:
    - ci_list: List of tuples with (lower, upper) bounds for the fit curve at different sigma levels.
    """
    ci_list = []
    sigmas_list = [None, None, None]
    try:
        if not hasattr(sigmas, "__len__"):
            sigmas_list[sigmas - 1] = sigmas
        else:
            for sigma in sigmas:
                sigmas_list[sigma - 1] = sigma
        for sigma in sigmas_list:
            if sigma is None:
                ci_list.append((None, None))
                continue
            # Calculate uncertainty at the given sigma level
            uncertainty = result.eval_uncertainty(sigma=sigma, x=xdata)
            best_fit = result.eval(x=xdata)

            lower = best_fit - uncertainty
            upper = best_fit + uncertainty

            ci_list.append((lower, upper))
    except Exception as e:
        import logging
        logging.error(f"Error calculating confidence intervals: {e}")
        for sigma in sigmas_list:
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

def print_fit_summary(result):
    print("Parameters and their errors:")
    for param, param_obj in result.params.items():
        print(f"{param:1}: {param_obj.value:6.4g} ± {param_obj.stderr:8.4g}  ({abs(param_obj.stderr/param_obj.value):.2%})")

    print(f"\nR^2: { result.rsquared}")

    report = result.fit_report()
    cov_start = report.find("Correlations")
    if cov_start != -1:
        print(report[cov_start:])
    else:
        print("No covariances found in the fit report.")
    #print(result.fit_report())

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
