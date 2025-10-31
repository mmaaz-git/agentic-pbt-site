import numpy as np
import numpy.ma as ma

# Create two masked arrays with identical unmasked values
x = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
y = ma.array([1.0, 999.0, 3.0], mask=[False, True, False])

print("Arrays:")
print(f"x = {x}")
print(f"y = {y}")
print()

print("Unmasked values comparison:")
print(f"x unmasked values: {ma.compressed(x)}")
print(f"y unmasked values: {ma.compressed(y)}")
print(f"Unmasked values are identical: {np.array_equal(ma.compressed(x), ma.compressed(y))}")
print()

print("allequal results:")
result_true = ma.allequal(x, y, fill_value=True)
print(f"ma.allequal(x, y, fill_value=True): {result_true}")

result_false = ma.allequal(x, y, fill_value=False)
print(f"ma.allequal(x, y, fill_value=False): {result_false}")
print()

print("Expected behavior:")
print("Since unmasked values are identical and masks are identical,")
print("the function should return True when comparing unmasked values.")
print("With fill_value=False, it incorrectly returns False without checking unmasked values.")