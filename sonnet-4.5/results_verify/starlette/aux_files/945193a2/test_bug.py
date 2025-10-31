import numpy as np
import numpy.lib.scimath as scimath

print("Testing the reported bug case:")
x = -2.0758172915594093e-87
p = -4.0

result = scimath.power(x, p)
print(f"scimath.power({x}, {p}) = {result}")
print(f"Result: {result}")
print(f"Imaginary part: {result.imag}")
print(f"Imaginary part is NaN: {np.isnan(result.imag)}")

print("\nComparison with other negative bases:")
for test_x in [-1.0, -1e-10, -1e-50]:
    test_result = scimath.power(test_x, -4.0)
    print(f"scimath.power({test_x}, -4.0) = {test_result}")

print("\nComparison with numpy.power (regular):")
try:
    regular_result = np.power(-2.0758172915594093e-87, -4.0)
    print(f"numpy.power({x}, {p}) = {regular_result}")
except Exception as e:
    print(f"numpy.power raised error: {e}")