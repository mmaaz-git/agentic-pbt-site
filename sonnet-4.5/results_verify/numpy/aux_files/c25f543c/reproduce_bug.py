import numpy as np
import numpy.ma as ma

# Example 1: From the bug report
x = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
y = ma.array([1.0, 999.0, 3.0], mask=[False, True, False])

print("Example 1:")
print(f"x values: {x.data}, mask: {x.mask}")
print(f"y values: {y.data}, mask: {y.mask}")
print(f"Unmasked x: {ma.compressed(x)}")
print(f"Unmasked y: {ma.compressed(y)}")
print(f"allequal(x, y, fill_value=False): {ma.allequal(x, y, fill_value=False)}")
print(f"allequal(x, y, fill_value=True): {ma.allequal(x, y, fill_value=True)}")
print()

# Example 2: The failing test case
x2 = ma.array([0., 0.], mask=[False, True])
y2 = ma.array([0., 0.], mask=[False, True])

print("Example 2 (failing test case):")
print(f"x2 values: {x2.data}, mask: {x2.mask}")
print(f"y2 values: {y2.data}, mask: {y2.mask}")
print(f"Unmasked x2: {ma.compressed(x2)}")
print(f"Unmasked y2: {ma.compressed(y2)}")
print(f"allequal(x2, y2, fill_value=False): {ma.allequal(x2, y2, fill_value=False)}")
print(f"allequal(x2, y2, fill_value=True): {ma.allequal(x2, y2, fill_value=True)}")
print()

# Example 3: No masked values
x3 = ma.array([1.0, 2.0, 3.0], mask=[False, False, False])
y3 = ma.array([1.0, 2.0, 3.0], mask=[False, False, False])

print("Example 3 (no masked values):")
print(f"x3 values: {x3.data}, mask: {x3.mask}")
print(f"y3 values: {y3.data}, mask: {y3.mask}")
print(f"allequal(x3, y3, fill_value=False): {ma.allequal(x3, y3, fill_value=False)}")
print(f"allequal(x3, y3, fill_value=True): {ma.allequal(x3, y3, fill_value=True)}")
print()

# Example 4: Different unmasked values
x4 = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
y4 = ma.array([1.0, 999.0, 4.0], mask=[False, True, False])  # Note: 4.0 instead of 3.0

print("Example 4 (different unmasked values):")
print(f"x4 values: {x4.data}, mask: {x4.mask}")
print(f"y4 values: {y4.data}, mask: {y4.mask}")
print(f"Unmasked x4: {ma.compressed(x4)}")
print(f"Unmasked y4: {ma.compressed(y4)}")
print(f"allequal(x4, y4, fill_value=False): {ma.allequal(x4, y4, fill_value=False)}")
print(f"allequal(x4, y4, fill_value=True): {ma.allequal(x4, y4, fill_value=True)}")
print()

# Example 5: Different masks
x5 = ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
y5 = ma.array([1.0, 2.0, 3.0], mask=[False, False, False])  # Different mask

print("Example 5 (different masks):")
print(f"x5 values: {x5.data}, mask: {x5.mask}")
print(f"y5 values: {y5.data}, mask: {y5.mask}")
print(f"allequal(x5, y5, fill_value=False): {ma.allequal(x5, y5, fill_value=False)}")
print(f"allequal(x5, y5, fill_value=True): {ma.allequal(x5, y5, fill_value=True)}")