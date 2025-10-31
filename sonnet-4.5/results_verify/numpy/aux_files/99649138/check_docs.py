import numpy.ma as ma
import numpy as np

# Get the docstring for mask_or
print("=" * 60)
print("numpy.ma.mask_or documentation:")
print("=" * 60)
print(ma.mask_or.__doc__)

print("\n" + "=" * 60)
print("Checking other mask functions for comparison:")
print("=" * 60)

# Check make_mask
print("\nnumpy.ma.make_mask docstring (excerpt):")
print("-" * 40)
print(ma.make_mask.__doc__[:500])

# Test make_mask with list
print("\nTesting make_mask with list:")
try:
    result = ma.make_mask([True, False, True])
    print(f"make_mask([True, False, True]) = {result}")
except Exception as e:
    print(f"Error: {e}")

# Check getmask
print("\n" + "=" * 60)
print("numpy.ma.getmask docstring (excerpt):")
print("-" * 40)
print(ma.getmask.__doc__[:500])

# Test getmask with list
print("\nTesting getmask with list:")
try:
    result = ma.getmask([1, 2, 3])
    print(f"getmask([1, 2, 3]) = {result}")
except Exception as e:
    print(f"Error: {e}")

# Check numpy's array_like convention
print("\n" + "=" * 60)
print("Testing NumPy's array_like convention:")
print("=" * 60)

# Test various numpy functions with lists
test_cases = [
    ("np.array([1, 2, 3])", lambda: np.array([1, 2, 3])),
    ("np.asarray([1, 2, 3])", lambda: np.asarray([1, 2, 3])),
    ("np.logical_or([True, False], [False, True])", lambda: np.logical_or([True, False], [False, True])),
    ("np.add([1, 2], [3, 4])", lambda: np.add([1, 2], [3, 4])),
]

for desc, func in test_cases:
    try:
        result = func()
        print(f"✓ {desc} works: {result}")
    except Exception as e:
        print(f"✗ {desc} fails: {e}")