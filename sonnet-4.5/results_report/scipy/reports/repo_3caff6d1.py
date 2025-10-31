from scipy.optimize.cython_optimize._zeros import full_output_example


def eval_polynomial(coeffs, x):
    a0, a1, a2, a3 = coeffs
    return a0 + a1*x + a2*x**2 + a3*x**3


args = (0.0, 1.0, 0.0, 0.0)
xa = 1e-100
xb = 1.0

f_xa = eval_polynomial(args, xa)
f_xb = eval_polynomial(args, xb)

print(f'f(xa={xa}) = {f_xa}')
print(f'f(xb={xb}) = {f_xb}')
print(f'Both positive - no root bracketed')
print()

result = full_output_example(args, xa, xb, 1e-6, 1e-6, 100)

print(f'error_num: {result["error_num"]} (expected -1 for sign error)')
print(f'root: {result["root"]}')
print()

assert result['error_num'] == -1, f'Expected sign error but got error_num={result["error_num"]}'