import dask.array as da
import numpy as np

print("Testing dask.array.eye with non-square matrix and chunks >= M")
print("=" * 60)

# Show that NumPy works correctly
print("\n1. NumPy baseline (works correctly):")
np_result = np.eye(2, M=3)
print(f"   np.eye(2, M=3) shape: {np_result.shape}")
print(f"   Result:\n{np_result}")

# Show the failing case
print("\n2. Dask failing case: da.eye(2, chunks=3, M=3, k=0)")
try:
    arr = da.eye(2, chunks=3, M=3, k=0)
    print(f"   Created array with shape {arr.shape}")
    print("   Attempting to compute...")
    result = arr.compute()
    print(f"   Success! Result:\n{result}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

# Show working cases for comparison
print("\n3. Dask working case 1: da.eye(2, chunks=2, M=3) [chunks < M]")
try:
    arr = da.eye(2, chunks=2, M=3)
    print(f"   Created array with shape {arr.shape}")
    result = arr.compute()
    print(f"   Success! Result:\n{result}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

print("\n4. Dask working case 2: da.eye(3, chunks=3, M=3) [square matrix]")
try:
    arr = da.eye(3, chunks=3, M=3)
    print(f"   Created array with shape {arr.shape}")
    result = arr.compute()
    print(f"   Success! Result:\n{result}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")