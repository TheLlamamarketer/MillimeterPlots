import numpy as np
import matplotlib.pyplot as plt


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
        power = power + (n - 1) if power > 0 else n
        return round(val, power), power

    else:
        power += n - 1
        factor = 10**power
        err_round = np.ceil(err * factor) / factor
        power = FirstSignificant(err_round)
        if intermed: power += 1
        return round(val, power), err_round, power


def support(xdata, ydata, yerr):  # Support function for main
    x2 = sum((x**2) / (dy**2) for x, dy in zip(xdata, yerr))
    x = sum(x / (dy**2) for x, dy in zip(xdata, yerr))
    y = sum(y / (dy**2) for y, dy in zip(ydata, yerr))
    xy = sum((x * y) / (dy**2) for x, y, dy in zip(xdata, ydata, yerr))
    one = sum(1 / (dy**2) for dy in yerr)

    return x, x2, y, xy, one

def main(xdata, ydata, yerr):

    (x, x2, y, xy, one) = support(xdata, ydata, yerr)  
    S = one * x2 - x**2  # determinant of the matrix
    a = (x2 * y - x * xy) / S 
    b = (one * xy - x * y) / S 
    da = np.sqrt(x2 / S)  
    db = np.sqrt(one / S)  

    zähler = sum(((y - a - b * x) / dy) ** 2 for x, y, dy in zip(xdata, ydata, yerr))  # Sum over deviation from fit
    nenner = sum(((y - y / one) / dy) ** 2 for y, dy in zip(ydata, yerr))  # Sum over deviation from mean value y/one
    R2 = (1 - zähler / nenner) * 100  #  R^2
    s2 = zähler / (len(xdata) - 2)  #  s^2

    return a, da, b, db, R2, s2


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
