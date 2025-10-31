import numpy as np
import numpy.lib.scimath as scimath

x = -1e-200
n = -2
result = scimath.power(x, n)

print(f'scimath.power({x}, {n}) = {result}')
print(f'Has NaN: {np.isnan(result)}')

# Let's also check what regular numpy.power does
try:
    regular_result = np.power(x, n)
    print(f'np.power({x}, {n}) = {regular_result}')
except Exception as e:
    print(f'np.power({x}, {n}) raised: {e}')

# Test with a positive base for comparison
x_pos = 1e-200
result_pos = scimath.power(x_pos, n)
print(f'\nscimath.power({x_pos}, {n}) = {result_pos}')
print(f'Has NaN: {np.isnan(result_pos)}')

# Test what happens with even positive power
x_neg = -1e-200
n_pos = 2
result_pos_power = scimath.power(x_neg, n_pos)
print(f'\nscimath.power({x_neg}, {n_pos}) = {result_pos_power}')
print(f'Has NaN: {np.isnan(result_pos_power)}')

# What about regular numpy with same?
regular_pos_power = np.power(x_neg, n_pos)
print(f'np.power({x_neg}, {n_pos}) = {regular_pos_power}')