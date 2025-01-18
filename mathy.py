import numpy as np
import sympy as sp

print()
print('-'*100)

data = {
    'lambda': np.array([1, 2, 3, 4, 5]),
    'y': np.array([2, 4, 6, 8, 10]),
    'z': np.array([3, 6, 9, 12, 15]),
    'dlambda': 0.1,
    'dy': 0.2,
    'dz': 0.3
}

def partial_derivatives(expr):
    symbols = expr.free_symbols
    partials = {symbol: sp.diff(expr, symbol) for symbol in symbols}
    return partials

def symbolic_error(expr, derivs):
    symbols = expr.free_symbols
    errors = {symbol: (derivs[symbol] * sp.Symbol(f'\\left( \\Delta {symbol} \\right)'))**2 for symbol in symbols}
    total_error = sp.sqrt(sum(errors.values()))
    return total_error

def calculate_error_value(symbolic_err, data, symbol_map):
    error_expr = symbolic_err
    for symbol in symbolic_err.free_symbols:
        symbol_str = str(symbol)
        if symbol_str.startswith('\\left( \\Delta '):
            base_symbol_str = symbol_str[13:-8]
            if base_symbol_str in symbol_map and f'd{symbol_map[base_symbol_str]}' in data:
                error_expr = error_expr.subs(symbol, data[f'd{symbol_map[base_symbol_str]}'])
        else:
            if symbol_str in symbol_map and symbol_map[symbol_str] in data:
                if isinstance(data[symbol_map[symbol_str]], np.ndarray):
                    error_expr = error_expr.subs(symbol, data[symbol_map[symbol_str]][0])
                else:
                    error_expr = error_expr.subs(symbol, data[symbol_map[symbol_str]])
    return error_expr.evalf()

def compute_and_print_error(expr, data, symbol_map):
    derivs = partial_derivatives(expr)
    symbolic_err = symbolic_error(expr, derivs)
    error_value = calculate_error_value(symbolic_err, data, symbol_map)
    
    print('\\begin{align*}')
    print('   ' + sp.latex(symbolic_err))
    print('\\end{align*}')
    print(f"Calculated error value: {error_value}")
    
    return error_value

lambda_, y, z = sp.symbols('\\lambda y z')
symbol_map = {'\\lambda': 'lambda', 'y': 'y', 'z': 'z'}
expr = lambda_**2 + y**2 + z**2 + lambda_*y*z

compute_and_print_error(expr, data, symbol_map)




