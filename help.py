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
    
def print_round_val(val, err=0, intermed=True):
    if err == 0:
        rounded_val, _ = round_val(val, intermed=intermed)
        return rounded_val
    else:
        rounded_val, rounded_err, _ = round_val(val, err, intermed=intermed)
        return f"{rounded_val} \\pm {rounded_err}"

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
    mask = ~np.isnan(xdata) & ~np.isnan(ydata) & np.isfinite(xdata) & np.isfinite(ydata)
    if yerr is not None:
        mask &= ~np.isnan(yerr) & np.isfinite(yerr)
        yerr = yerr[mask]
    xdata = xdata[mask]
    ydata = ydata[mask]

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
    Fit data to a specified model: linear, quadratic, or exponential.
    - constraints: Dictionary of parameter constraints, e.g., {"a": 0}.
    """
    if yerr is not None and not hasattr(yerr, "__len__"):
        yerr = np.array([yerr])

    # Remove NaN and infinite values
    mask = ~np.isnan(xdata) & ~np.isnan(ydata) & np.isfinite(xdata) & np.isfinite(ydata)
    if yerr is not None:
        mask &= ~np.isnan(yerr) & np.isfinite(yerr)
    xdata, ydata = xdata[mask], ydata[mask]
    if yerr is not None:
        yerr = yerr[mask]

    # Define models
    if model == "linear":
        def model_func(x, a, b): return a + b * x
    elif model == "quadratic":
        def model_func(x, a, b, c): return a + b * x + c * x**2
    elif model == "exponential":
        def model_func(x, a, b, c): return a * np.exp(b * x) + c
    elif model == "exponential_decay":
        def model_func(x, a, b, c): return a * (1 - np.exp(-b * x + c))
    elif model == "gaussian":
        def model_func(x, a, b, c, d): return a * np.exp(-((x - b) / c)**2 /2) + d
    else:
        raise ValueError(f"Unsupported model: {model}")

    mode_func = Model(model_func)

    # Set initial parameters
    if model == "exponential":
        params = mode_func.make_params(a=np.max(ydata), b=0.2, c=np.min(ydata))
    elif model == "exponential_decay":
        params = mode_func.make_params(a=np.max(ydata), b=0.7, c=0)
    elif model == "gaussian":
        params = mode_func.make_params(a=np.max(ydata), b=np.mean(xdata), c=1, d=np.min(ydata))
    else:
        params = mode_func.make_params(a=0, b=1, c=1)

    if constraints:
        for param, value in constraints.items():
            if value is not None:
                if isinstance(value, dict):  # Allow setting min/max
                    params[param].set(**value)
                else:
                    params[param].set(value=value, vary=False)

    # Perform the fit
    if yerr is not None:
        result = mode_func.fit(ydata, params, x=xdata, weights=1.0 / yerr**2)
    else:
        result = mode_func.fit(ydata, params, x=xdata)

    #print(result.fit_report())

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
