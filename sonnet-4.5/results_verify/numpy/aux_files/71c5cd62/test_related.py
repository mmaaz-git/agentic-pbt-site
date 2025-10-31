import numpy as np

print("Testing empty record array creation:")
try:
    empty_rec = np.recarray(shape=(0,), dtype=[('x', int), ('y', int)])
    print(f"Success! Created empty recarray: shape={empty_rec.shape}, len={len(empty_rec)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting np.rec.fromarrays with empty lists:")
try:
    rec = np.rec.fromarrays([[], []], names='a,b')
    print(f"Success! Created record array from empty arrays: shape={rec.shape}, len={len(rec)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting np.rec.fromarrays with non-empty lists:")
try:
    rec = np.rec.fromarrays([[1, 2, 3], [4.0, 5.0, 6.0]], names='a,b')
    print(f"Success! Created record array: shape={rec.shape}, len={len(rec)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting np.array with empty list and structured dtype:")
try:
    arr = np.array([], dtype=[('x', int), ('y', float)])
    print(f"Success! Created empty structured array: shape={arr.shape}, len={len(arr)}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")