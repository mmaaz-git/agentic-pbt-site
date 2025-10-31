import numpy as np
import numpy.ma as ma

# Test to understand what the semantic meaning should be

print("Understanding the semantic meaning of fill_value parameter:")
print("=" * 60)

# Case A: Two arrays with same mask, identical values
a = ma.array([1, 2, 3], mask=[False, True, False])
b = ma.array([1, 2, 3], mask=[False, True, False])
print("\nCase A: Identical arrays with same mask")
print(f"  Arrays are completely identical")
print(f"  Expected with fill_value=True: True (masked treated as equal)")
print(f"  Expected with fill_value=False: ? (masked treated as not equal)")
print(f"  Actual with fill_value=False: {ma.allequal(a, b, fill_value=False)}")

# Case B: Two arrays, different at masked position only
a = ma.array([1, 2, 3], mask=[False, True, False])
b = ma.array([1, 999, 3], mask=[False, True, False])
print("\nCase B: Different only at masked position")
print(f"  Unmasked values: identical")
print(f"  Masked values: different (but both masked)")
print(f"  Expected with fill_value=True: True (ignore masked difference)")
print(f"  Expected with fill_value=False: ? (how to handle masked?)")
print(f"  Actual with fill_value=False: {ma.allequal(a, b, fill_value=False)}")

# Case C: Different masks
a = ma.array([1, 2, 3], mask=[False, True, False])
b = ma.array([1, 2, 3], mask=[False, False, True])
print("\nCase C: Different masks, same underlying data")
print(f"  Where both unmasked: values match")
print(f"  Masks differ at positions 1 and 2")
print(f"  Expected with fill_value=True: True (masked=equal)")
print(f"  Expected with fill_value=False: ? (masked≠unmasked?)")
print(f"  Actual with fill_value=False: {ma.allequal(a, b, fill_value=False)}")

print("\n" + "=" * 60)
print("Interpretation Options for fill_value=False:")
print("1. Current: Return False if ANY masked values exist")
print("2. Option A: Masked positions are 'not equal' to anything")
print("3. Option B: Masked positions in both arrays are still equal to each other")
print("4. Option C: Only compare unmasked positions")

# Let's check what numpy.array_equal does with NaN
print("\n" + "=" * 60)
print("How does numpy handle NaN in array_equal?")
x = np.array([1.0, np.nan, 3.0])
y = np.array([1.0, np.nan, 3.0])
print(f"array_equal with NaN: {np.array_equal(x, y)}")  # False - NaN != NaN
print(f"allclose with NaN: {np.allclose(x, y, equal_nan=False)}")  # False
print(f"allclose with NaN (equal_nan=True): {np.allclose(x, y, equal_nan=True)}")  # True