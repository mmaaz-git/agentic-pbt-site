import scipy.datasets._utils as utils
import os

cache_path = "/tmp/nonexistent_scipy_cache_xyz_test_12345"

# Make sure the cache directory doesn't exist
if os.path.exists(cache_path):
    import shutil
    shutil.rmtree(cache_path)

print("Testing with non-existent cache directory:")
print(f"Cache path: {cache_path}")
print(f"Cache exists: {os.path.exists(cache_path)}")
print()

print("Test 1: Passing integer 42")
try:
    utils._clear_cache(datasets=42, cache_dir=cache_path)
    print("No exception raised - function returned normally")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")
print()

print("Test 2: Passing string 'not a callable'")
try:
    utils._clear_cache(datasets="not a callable", cache_dir=cache_path)
    print("No exception raised - function returned normally")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")
print()

print("Test 3: Passing list of integers [1, 2, 3]")
try:
    utils._clear_cache(datasets=[1, 2, 3], cache_dir=cache_path)
    print("No exception raised - function returned normally")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")
print()

# Now test with an existing cache directory to see different behavior
print("Creating cache directory and testing again...")
os.makedirs(cache_path)
print(f"Cache exists: {os.path.exists(cache_path)}")
print()

print("Test 4: Passing integer 42 with existing cache")
try:
    utils._clear_cache(datasets=42, cache_dir=cache_path)
    print("No exception raised - function returned normally")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")
print()

# Clean up
if os.path.exists(cache_path):
    os.rmdir(cache_path)