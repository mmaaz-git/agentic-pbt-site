import numpy.ma as ma
import numpy as np

# Reproduce the existing test cases
print("Test cases from numpy test suite:")
print("=" * 60)

# Test case 1
x = ma.array([1, 2, 3], mask=[0, 0, 0])
y = ma.array([1, 2, 3], mask=[1, 0, 0])
print("\nTest 1:")
print(f"x = array([1, 2, 3], mask=[0, 0, 0])")
print(f"y = array([1, 2, 3], mask=[1, 0, 0])")
print(f"x unmasked: {ma.compressed(x)}")
print(f"y unmasked: {ma.compressed(y)}")
print(f"allequal(x, y): {ma.allequal(x, y)}")
print(f"allequal(x, y, fill_value=False): {ma.allequal(x, y, fill_value=False)}")
print(f"Test expects: allequal(x, y) = True")
print(f"Test expects: allequal(x, y, fill_value=False) = False")

# Test case 2
print("\nTest 2:")
print(f"x = array([1, 2, 3], mask=[0, 0, 0]) - no masked values")
print(f"allequal(x, x): {ma.allequal(x, x)}")
print(f"allequal(x, x, fill_value=False): {ma.allequal(x, x, fill_value=False)}")
print(f"Test expects: allequal(x, x) = True")
print(f"Test expects: allequal(x, x, fill_value=False) = True")

# Test case 3
print("\nTest 3:")
print(f"y = array([1, 2, 3], mask=[1, 0, 0]) - has masked value")
print(f"allequal(y, y): {ma.allequal(y, y)}")
print(f"allequal(y, y, fill_value=False): {ma.allequal(y, y, fill_value=False)}")
print(f"Test expects: allequal(y, y) = True")
print(f"Test expects: allequal(y, y, fill_value=False) = False")

print("\n" + "=" * 60)
print("ANALYSIS:")
print("The existing test expects that allequal(y, y, fill_value=False) = False")
print("where y is an array with some masked values.")
print("This suggests the intended behavior is: if fill_value=False,")
print("masked values are NOT equal to anything, even themselves.")