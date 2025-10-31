import numpy as np

# Test various dtypes
test_cases = [
    np.dtype('i4'),  # Simple dtype
    np.dtype(('i4', (1,))),  # Sub-array dtype with shape (1,)
    np.dtype(('i4', (2, 3))),  # Sub-array dtype with shape (2, 3)
    np.dtype([('x', 'i4'), ('y', 'f8')])  # Structured dtype
]

for dtype in test_cases:
    print(f"\nDtype: {dtype}")
    print(f"  dtype.names: {dtype.names}")
    print(f"  dtype.shape: {dtype.shape}")
    print(f"  dtype.str: {dtype.str}")

    if dtype.shape:
        print(f"  dtype.base: {dtype.base}")
        print(f"  dtype.base.str: {dtype.base.str}")

    # What dtype_to_descr returns
    descr = np.lib.format.dtype_to_descr(dtype)
    print(f"  dtype_to_descr returns: {repr(descr)}")

    # Try to recreate
    try:
        restored = np.lib.format.descr_to_dtype(descr)
        print(f"  Restored dtype: {restored}")
        print(f"  Round-trip successful: {restored == dtype}")
    except Exception as e:
        print(f"  ERROR restoring: {e}")