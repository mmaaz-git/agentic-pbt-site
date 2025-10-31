import numpy as np
import numpy.ma as ma

print("=== Test Case 1: Simple reproduction ===")
x = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
y = ma.array([1.0, 999.0, 3.0], mask=[False, True, False])

print(f"x = {x}")
print(f"y = {y}")
print(f"x.mask = {x.mask}")
print(f"y.mask = {y.mask}")

compressed_x = ma.compressed(x)
compressed_y = ma.compressed(y)
print(f"compressed(x) = {compressed_x}")
print(f"compressed(y) = {compressed_y}")
print(f"Unmasked values are identical: {np.array_equal(compressed_x, compressed_y)}")

result_false = ma.allequal(x, y, fill_value=False)
result_true = ma.allequal(x, y, fill_value=True)
print(f"allequal(x, y, fill_value=False): {result_false}")
print(f"allequal(x, y, fill_value=True): {result_true}")

print("\n=== Test Case 2: Failing input from bug report ===")
data = np.array([0., 0.])
mask = np.array([False, True])

x2 = ma.array(data, mask=mask)
y2 = ma.array(data.copy(), mask=mask.copy())

print(f"x2 = {x2}")
print(f"y2 = {y2}")
print(f"x2.mask = {x2.mask}")
print(f"y2.mask = {y2.mask}")

unmasked_equal = np.array_equal(data[~mask], data[~mask])
print(f"Unmasked values are identical: {unmasked_equal}")

result2_false = ma.allequal(x2, y2, fill_value=False)
result2_true = ma.allequal(x2, y2, fill_value=True)
print(f"allequal(x2, y2, fill_value=False): {result2_false}")
print(f"allequal(x2, y2, fill_value=True): {result2_true}")

print("\n=== Test Case 3: No masked values ===")
x3 = ma.array([1.0, 2.0, 3.0])
y3 = ma.array([1.0, 2.0, 3.0])

print(f"x3 = {x3}")
print(f"y3 = {y3}")
result3_false = ma.allequal(x3, y3, fill_value=False)
result3_true = ma.allequal(x3, y3, fill_value=True)
print(f"allequal(x3, y3, fill_value=False): {result3_false}")
print(f"allequal(x3, y3, fill_value=True): {result3_true}")

print("\n=== Test Case 4: Different masks ===")
x4 = ma.array([1.0, 2.0, 3.0], mask=[True, False, False])
y4 = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])

print(f"x4 = {x4}")
print(f"y4 = {y4}")
result4_false = ma.allequal(x4, y4, fill_value=False)
result4_true = ma.allequal(x4, y4, fill_value=True)
print(f"allequal(x4, y4, fill_value=False): {result4_false}")
print(f"allequal(x4, y4, fill_value=True): {result4_true}")