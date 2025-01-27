from math import e
import mathy as sp
from sympy.abc import n, t, x
from mathy import oo
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy.utilities.lambdify import lambdify

n = sp.symbols('n', integer=True, nonnegative=True)

def compute_borel_sum(a_n_expr, x_exp, x_value=None):
    """
    Compute the Borel sum of a series given the general term a_n and exponent of x.

    Parameters:
    a_n_expr: SymPy expression for a_n in terms of n.
    x_exp: SymPy expression for the exponent of x in terms of n (e.g., n, 2*n).
    x_value: Numerical value of x (optional).

    Returns:
    A tuple (B_t, S_B_x), where:
    - B_t is the Borel transform B(t).
    - S_B_x is the Borel sum S_B(x), symbolic or numerical depending on x_value.
    """
    # Define symbols
    n = sp.symbols('n', integer=True, nonnegative=True)
    t = sp.symbols('t', real=True, positive=True)
    x_sym = sp.symbols('x', real=True)
    
    # Determine the exponent multiplier k from x_exp
    k = sp.simplify(x_exp / n)
    if not k.is_number:
        raise ValueError("x_exp must be a linear function of n.")
    k = float(k)
    
    # Compute the Borel transform B(t)
    a_n = a_n_expr
    term = (a_n / sp.factorial(n)) * t**n
    B_t = sp.summation(term, (n, 0, oo))
    B_t = sp.simplify(B_t)
    B_txk = B_t.subs(t, t * x_sym**k)
    integrand = sp.exp(-t) * B_txk

    if x_value == 0:
        return B_t, 0  # Handle x=0 case directly
    
    if x_value is not None:
        # Substitute x_value and check integrand validity
        integrand_numeric = integrand.subs(x_sym, x_value)
        if integrand_numeric.has(sp.zoo) or integrand_numeric.has(sp.nan) or integrand_numeric.has(sp.oo):
            print(f"Integrand invalid for x = {x_value}")
            return B_t, np.nan  # Return NaN if integrand is invalid

        # Convert to numerical function
        integrand_func = sp.lambdify(t, integrand_numeric, modules=['numpy'])
        try:
            # Numerical integration
            integral_numeric, error = quad(lambda t: integrand_func(t), 0, np.inf, limit=1000)
        except Exception as e:
            print(f"Integration failed for x = {x_value} with error: {e}")
            return B_t, np.nan  # Return NaN if integration fails
        return B_t, integral_numeric
    else:
        # Attempt analytical integration
        S_B_x = sp.integrate(integrand, (t, 0, oo))
        return B_t, S_B_x
    

borel = False
xmin =-15
xmax = 20
x_values = np.linspace(xmin, xmax, 1000)
a_n_expr = 1/sp.factorial(n)
x_exp = n
exact_values = np.exp(x_values)


#a_n_expr = (-1)**n
#x_exp = 2 * n  
#exact_values = 1 / (1 + x_values**2)

#a_n_expr = sp.factorial(n)
#x_exp = n
#exact_values = None


#a_n_expr = 1
#x_exp = n
#exact_values = 1/(1-x_values)


# Define max_terms_list for partial sums
#max_terms_list = [5, 10, 20, 200] 

max_terms_list = [1, 2, 3, 4, 5, 20, 30]

# Compute partial sums of the Taylor series
def taylor_series_partial_sums(a_n, n_var, x_exp, x_var, x_values, max_terms_list):
    partial_sums_list = []
    for N in max_terms_list:
        # Compute the partial sum symbolically
        partial_sum_expr = sp.Sum(a_n * x_var**x_exp, (n_var, 0, N)).doit()
        partial_sum_func = lambdify(x_var, partial_sum_expr, 'numpy')

        try:
            partial_sums = partial_sum_func(x_values)
        except OverflowError:
            partial_sums = np.full_like(x_values, np.inf)
        partial_sums_list.append(partial_sums)
    return partial_sums_list

partial_sums_list = taylor_series_partial_sums(a_n_expr, n, x_exp, x, x_values, max_terms_list)


plt.figure(figsize=(25.5, 9))


if exact_values is not None:
    plt.plot(x_values, exact_values, label=f'f(x) = $e^x$', color='black', linewidth=3)

if borel:
    borel_sums = []
    x_vals = []
    failure_count = 0
    max_failures = 10  # Stop after 10 consecutive failures

    for i, x_val in enumerate(x_values):
        _, S_B_x_val = compute_borel_sum(a_n_expr, x_exp, x_value=x_val)
        if np.isnan(S_B_x_val):  # Check if the value is NaN
            failure_count += 1
        else:
            failure_count = 0  # Reset failure count if successful
        
        # Stop if too many failures occur
        if failure_count >= max_failures:
            print(f"Stopping early: too many consecutive failures at x = {x_val}")
            break

        borel_sums.append(S_B_x_val)
        x_vals.append(x_val)
        # Print progress
        if (i + 1) % (len(x_values) // 100) == 0:
            print(f"Progress: {(i + 1) / len(x_values) * 100:.0f}%")

    borel_sums = np.array(borel_sums)

    plt.plot(x_vals, borel_sums, label='Borel Summe', color='red', linestyle='--', linewidth=5)

for partial_sums, N in zip(partial_sums_list, max_terms_list):
    plt.plot(x_values, partial_sums, label=f'Taylor Reihe (N={N})', alpha=0.7)


#plt.title('$f(x)= \\sum_n(n!x^n)$')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend()
plt.grid(True)
# have the ratio of x and y be the same
plt.tight_layout()
plt.subplots_adjust(left=0.06) 
plt.subplots_adjust(right=0.98)
plt.ylim(-2, 50)
plt.xlim(xmin, xmax)
plt.show()
