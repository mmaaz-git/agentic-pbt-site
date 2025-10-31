import numpy.ma as ma
import numpy as np

def mask_or_fixed(m1, m2, copy=False, shrink=True):
    """Fixed version of mask_or"""
    nomask = ma.nomask
    MaskType = ma.MaskType
    make_mask = ma.make_mask
    is_mask = ma.is_mask
    umath = np.core.umath

    if (m1 is nomask) or (m1 is False):
        dtype = getattr(m2, 'dtype', MaskType)
        return make_mask(m2, copy=copy, shrink=shrink, dtype=dtype)
    if (m2 is nomask) or (m2 is False):
        dtype = getattr(m1, 'dtype', MaskType)
        return make_mask(m1, copy=copy, shrink=shrink, dtype=dtype)
    if m1 is m2 and is_mask(m1):
        return ma.core._shrink_mask(m1) if shrink else m1
    (dtype1, dtype2) = (getattr(m1, 'dtype', None), getattr(m2, 'dtype', None))
    if dtype1 != dtype2:
        raise ValueError(f"Incompatible dtypes '{dtype1}'<>'{dtype2}'")
    # FIXED: Check if dtype1 is not None before accessing .names
    if dtype1 is not None and dtype1.names is not None:
        # Allocate an output mask array with the properly broadcast shape.
        newmask = np.empty(np.broadcast(m1, m2).shape, dtype1)
        ma.core._recursive_mask_or(m1, m2, newmask)
        return newmask
    return make_mask(umath.logical_or(m1, m2), copy=copy, shrink=shrink)

# Test the fix
print("Testing fixed version with plain lists:")
m1 = [False]
m2 = [False]
try:
    result = mask_or_fixed(m1, m2)
    print(f"Success! Result: {result}")
    print(f"Result type: {type(result)}")
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()

# Test with other list cases
print("\nTesting with [True, False] lists:")
try:
    result = mask_or_fixed([True, False], [False, True])
    print(f"Result: {result}")
except Exception as e:
    print(f"Failed: {e}")

# Verify it still works with numpy arrays
print("\nTesting with numpy arrays (should still work):")
try:
    result = mask_or_fixed(np.array([False]), np.array([False]))
    print(f"Result: {result}")
except Exception as e:
    print(f"Failed: {e}")