import numpy as np

print("Testing numpy's putmask behavior with length mismatch:")
print("=" * 50)

# Test 1: Single value repeated
values = np.arange(10)
mask = np.ones(10, dtype=bool)
new = np.array([999])

print(f"Test 1: Single element array")
print(f"values: {values}")
print(f"mask: {mask} (sum={mask.sum()})")
print(f"new: {new}")

np.putmask(values, mask, new)
print(f"Result: {values}")
print("Behavior: Single value is repeated")

print("\n" + "=" * 50)

# Test 2: Multiple values but fewer than mask
values2 = np.arange(10)
mask2 = np.ones(10, dtype=bool)
new2 = np.array([100, 200, 300])

print(f"Test 2: Array with 3 elements, mask has 10 True")
print(f"values: {values2}")
print(f"mask: {mask2} (sum={mask2.sum()})")
print(f"new: {new2}")

np.putmask(values2, mask2, new2)
print(f"Result: {values2}")
print("Behavior: Values are repeated in cycle")

print("\n" + "=" * 50)

# Test 3: What np.place does
values3 = np.arange(10)
mask3 = np.ones(10, dtype=bool)
new3 = np.array([100, 200, 300])

print(f"Test 3: Using np.place instead")
print(f"values: {values3}")
print(f"mask: {mask3} (sum={mask3.sum()})")
print(f"new: {new3}")

try:
    np.place(values3, mask3, new3)
    print(f"Result: {values3}")
except ValueError as e:
    print(f"Error: {e}")
    print("Behavior: np.place raises error on length mismatch!")