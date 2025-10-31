import numpy as np

# Create a structured dtype with two fields
dtype = np.dtype([('x', np.int32), ('y', np.float64)])

# Create an array with this structured dtype
arr = np.array([(1, 1.5), (2, 2.5), (3, 3.5)], dtype=dtype)

print("=" * 60)
print("Testing numpy.ctypeslib.as_ctypes with structured array")
print("=" * 60)
print(f"\nArray created:")
print(f"  data: {arr}")
print(f"  dtype: {arr.dtype}")
print(f"  shape: {arr.shape}")

# Show what the array interface typestr looks like for structured arrays
print(f"\nArray interface info:")
ai = arr.__array_interface__
print(f"  typestr: {ai['typestr']}")
print(f"  shape: {ai['shape']}")

print("\n" + "-" * 60)
print("Attempting np.ctypeslib.as_ctypes(arr):")
print("-" * 60)
try:
    ctypes_obj = np.ctypeslib.as_ctypes(arr)
    print(f"Success! Result: {ctypes_obj}")
    print(f"Type: {type(ctypes_obj)}")
except NotImplementedError as e:
    print(f"ERROR - NotImplementedError: {e}")

    print("\n" + "-" * 60)
    print("However, as_ctypes_type CAN handle the dtype directly:")
    print("-" * 60)
    try:
        ctype = np.ctypeslib.as_ctypes_type(arr.dtype)
        print(f"  np.ctypeslib.as_ctypes_type(arr.dtype) = {ctype}")
        print(f"  This is a ctypes.Structure with the correct fields")
    except Exception as e2:
        print(f"  Error: {e2}")

    print("\n" + "-" * 60)
    print("The problem: as_ctypes passes typestr instead of dtype")
    print("-" * 60)
    print(f"  as_ctypes calls: as_ctypes_type(ai['typestr'])")
    print(f"  ai['typestr'] = '{ai['typestr']}' (void type, loses field info)")
    print(f"  Should call: as_ctypes_type(obj.dtype)")
    print(f"  obj.dtype = {arr.dtype} (preserves field info)")