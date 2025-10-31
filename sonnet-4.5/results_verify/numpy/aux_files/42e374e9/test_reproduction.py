import numpy as np
import numpy.ma as ma

print("Testing fill value functions with dtype classes vs instances:\n")

for func_name, func in [('default_fill_value', ma.default_fill_value),
                         ('maximum_fill_value', ma.maximum_fill_value),
                         ('minimum_fill_value', ma.minimum_fill_value)]:
    print(f"Testing {func_name}:")
    print("-" * 40)

    # Test with dtype class
    try:
        fill = func(np.int32)
        print(f"  {func_name}(np.int32): {fill}")
    except AttributeError as e:
        print(f"  {func_name}(np.int32): FAILS - AttributeError: {e}")
    except Exception as e:
        print(f"  {func_name}(np.int32): FAILS - {type(e).__name__}: {e}")

    # Test with dtype instance
    try:
        fill_instance = func(np.dtype('int32'))
        print(f"  {func_name}(np.dtype('int32')): {fill_instance}")
    except Exception as e:
        print(f"  {func_name}(np.dtype('int32')): FAILS - {type(e).__name__}: {e}")

    # Test with string
    try:
        fill_string = func('int32')
        print(f"  {func_name}('int32'): {fill_string}")
    except Exception as e:
        print(f"  {func_name}('int32'): FAILS - {type(e).__name__}: {e}")

    print()

# Let's also test other dtype classes
print("\nTesting with various dtype classes:")
print("-" * 40)
dtype_classes = [np.int8, np.int16, np.int32, np.int64,
                 np.float16, np.float32, np.float64,
                 np.complex64, np.complex128]

for dtype_class in dtype_classes:
    try:
        result = ma.default_fill_value(dtype_class)
        print(f"ma.default_fill_value({dtype_class.__name__}): {result}")
    except AttributeError:
        print(f"ma.default_fill_value({dtype_class.__name__}): FAILS with AttributeError")
    except Exception as e:
        print(f"ma.default_fill_value({dtype_class.__name__}): FAILS - {type(e).__name__}")

# Let's understand what np.int32.dtype actually is
print("\n\nUnderstanding np.int32.dtype:")
print("-" * 40)
print(f"np.int32: {np.int32}")
print(f"type(np.int32): {type(np.int32)}")
print(f"np.int32.dtype: {np.int32.dtype}")
print(f"type(np.int32.dtype): {type(np.int32.dtype)}")
print(f"hasattr(np.int32, 'dtype'): {hasattr(np.int32, 'dtype')}")

# Check if np.int32 is a type and subclass of np.generic
print(f"\nisinstance(np.int32, type): {isinstance(np.int32, type)}")
print(f"issubclass(np.int32, np.generic): {issubclass(np.int32, np.generic)}")

# Compare with an instance
inst = np.int32(5)
print(f"\nnp.int32(5): {inst}")
print(f"type(np.int32(5)): {type(inst)}")
print(f"np.int32(5).dtype: {inst.dtype}")
print(f"type(np.int32(5).dtype): {type(inst.dtype)}")

# Test that NumPy functions normally accept dtype classes
print("\n\nVerifying NumPy's general behavior with dtype classes:")
print("-" * 40)
print(f"np.array([1,2,3], dtype=np.int32): {np.array([1,2,3], dtype=np.int32).dtype}")
print(f"np.zeros(3, dtype=np.float64): {np.zeros(3, dtype=np.float64).dtype}")
print(f"np.ones(2, dtype=np.complex128): {np.ones(2, dtype=np.complex128).dtype}")