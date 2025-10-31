import numpy as np
import numpy.lib.scimath as scimath

print("Testing specific failing case from bug report:")
x = -2.0758172915594093e-87
p = -4.0

result = scimath.power(x, p)
print(f"Input: x={x}, p={p}")
print(f"Result: {result}")

if np.isnan(result.imag):
    print(f"✗ Test FAILED: Imaginary part is NaN")
else:
    print(f"✓ Test PASSED: Imaginary part is not NaN")

print("\nTesting edge cases around very small negative numbers:")
test_cases = [
    (-1e-80, -4.0),
    (-1e-85, -4.0),
    (-1e-86, -4.0),
    (-1e-87, -4.0),
    (-1e-88, -4.0),
    (-1e-100, -4.0),
    (-1e-150, -4.0),
    (-1e-200, -4.0),
]

for x, p in test_cases:
    result = scimath.power(x, p)
    is_nan = np.isnan(result.imag) if np.iscomplex(result) else False
    print(f"power({x:12.2e}, {p}) = {result:20s} | Imag is NaN: {is_nan}")