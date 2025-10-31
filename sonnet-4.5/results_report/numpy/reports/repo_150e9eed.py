import numpy as np
import numpy.lib.scimath as scimath

# Test case from the bug report
x = -9.499558537778752e-188
n = -2
result = scimath.power(x, n)

print(f'scimath.power({x}, {n}) = {result}')
print(f'Has NaN in result: {np.isnan(result)}')
print(f'Result type: {type(result)}')
print(f'Result dtype: {result.dtype if hasattr(result, "dtype") else "N/A"}')

# Let's also test with simpler values
print("\nTesting with -1e-200:")
x2 = -1e-200
result2 = scimath.power(x2, -2)
print(f'scimath.power({x2}, {n}) = {result2}')
print(f'Has NaN: {np.isnan(result2)}')

# For comparison, let's see what regular numpy.power does
print("\nComparing with numpy.power:")
try:
    numpy_result = np.power(x2, -2)
    print(f'np.power({x2}, {n}) = {numpy_result}')
except Exception as e:
    print(f'np.power raised exception: {e}')

# Let's test the mathematical expectation
print("\nMathematical expectation:")
print(f"(-1e-200)^(-2) = 1/((-1e-200)^2) = 1/(1e-400) = 1e+400 = inf")
print(f"Since negative^even = positive, result should be positive infinity")

# Let's trace through what's happening step by step
print("\nStep-by-step trace:")
print(f"1. Input: x={x2}, p={n}")

# Convert to complex as scimath.power does
x_complex = np.asarray(x2, dtype=complex)
print(f"2. After _fix_real_lt_zero: x={x_complex}")

# Square it
squared = x_complex ** 2
print(f"3. x^2 = {squared}")

# Take reciprocal
if squared != 0:
    reciprocal = 1 / squared
    print(f"4. 1/(x^2) = {reciprocal}")
else:
    print(f"4. 1/(x^2) = division by {squared}")

# Check if the issue happens with other small negative values
print("\nTesting threshold for NaN occurrence:")
test_values = [-1e-150, -1e-155, -1e-160, -1e-170, -1e-180, -1e-190, -1e-200]
for val in test_values:
    res = scimath.power(val, -2)
    has_nan = np.isnan(res)
    print(f"scimath.power({val:e}, -2) = {res}, has NaN: {has_nan}")