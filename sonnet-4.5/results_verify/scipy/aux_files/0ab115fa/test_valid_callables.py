import scipy.datasets
import os

# Test with valid callables
cache_path = "/tmp/test_valid_cache_xyz_12345"

# Ensure directory doesn't exist
if os.path.exists(cache_path):
    import shutil
    shutil.rmtree(cache_path)

print("Testing with valid callables when cache doesn't exist:")

# Test 1: Pass a valid dataset method
print("\nTest 1: Passing scipy.datasets.ascent (valid callable)")
try:
    scipy.datasets._utils._clear_cache(datasets=scipy.datasets.ascent, cache_dir=cache_path)
    print("Success - function returned normally")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

# Test 2: Pass a list of valid dataset methods
print("\nTest 2: Passing list of valid callables")
try:
    scipy.datasets._utils._clear_cache(datasets=[scipy.datasets.ascent, scipy.datasets.face], cache_dir=cache_path)
    print("Success - function returned normally")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

# Test 3: Pass None (should clear all)
print("\nTest 3: Passing None")
try:
    scipy.datasets._utils._clear_cache(datasets=None, cache_dir=cache_path)
    print("Success - function returned normally")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

# Test 4: Pass a random callable that's not a dataset method
print("\nTest 4: Passing random callable (not a dataset method)")
def my_function():
    pass

try:
    scipy.datasets._utils._clear_cache(datasets=my_function, cache_dir=cache_path)
    print("Success - function returned normally")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")