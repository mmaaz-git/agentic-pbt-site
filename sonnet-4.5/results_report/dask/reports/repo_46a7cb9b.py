import numpy as np
import numpy.ma as ma

# Test with dtype class (should fail)
print("Testing with dtype class np.int32:")
print("=" * 50)

for func_name, func in [('default_fill_value', ma.default_fill_value),
                         ('maximum_fill_value', ma.maximum_fill_value),
                         ('minimum_fill_value', ma.minimum_fill_value)]:
    try:
        fill = func(np.int32)
        print(f"{func_name}(np.int32): {fill}")
    except AttributeError as e:
        print(f"{func_name}(np.int32): FAILS - AttributeError: {e}")

print("\nTesting with dtype instance np.dtype('int32'):")
print("=" * 50)

for func_name, func in [('default_fill_value', ma.default_fill_value),
                         ('maximum_fill_value', ma.maximum_fill_value),
                         ('minimum_fill_value', ma.minimum_fill_value)]:
    try:
        fill = func(np.dtype('int32'))
        print(f"{func_name}(np.dtype('int32')): {fill}")
    except Exception as e:
        print(f"{func_name}(np.dtype('int32')): FAILS - {type(e).__name__}: {e}")