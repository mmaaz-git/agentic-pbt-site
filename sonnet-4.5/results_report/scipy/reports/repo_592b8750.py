import scipy.datasets
import os
import shutil
import platformdirs

# Ensure cache directory doesn't exist for the first tests
cache_dir = platformdirs.user_cache_dir("scipy-data")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

print("=" * 60)
print("Test 1: Passing non-callable string when cache doesn't exist")
print("=" * 60)
try:
    scipy.datasets.clear_cache("not_callable")
    print("No exception raised - function returned successfully")
except (AssertionError, TypeError, ValueError) as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Test 2: Passing invalid callable when cache doesn't exist")
print("=" * 60)
def invalid_dataset():
    """This is not a valid scipy dataset function"""
    pass

try:
    scipy.datasets.clear_cache(invalid_dataset)
    print("No exception raised - function returned successfully")
except (AssertionError, TypeError, ValueError) as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

# Now create cache directory to test the same inputs when cache exists
print("\n" + "=" * 60)
print("Creating cache directory and testing same inputs...")
print("=" * 60)
os.makedirs(cache_dir, exist_ok=True)

print("\n" + "=" * 60)
print("Test 3: Passing non-callable string when cache EXISTS")
print("=" * 60)
try:
    scipy.datasets.clear_cache("not_callable")
    print("No exception raised - function returned successfully")
except (AssertionError, TypeError, ValueError) as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Test 4: Passing invalid callable when cache EXISTS")
print("=" * 60)
try:
    scipy.datasets.clear_cache(invalid_dataset)
    print("No exception raised - function returned successfully")
except (AssertionError, TypeError, ValueError) as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

# Clean up
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)