import os
import shutil
from scipy.datasets._utils import _clear_cache

def invalid_dataset():
    pass

invalid_dataset.__name__ = "not_a_dataset"

cache_dir = "/tmp/scipy_test_existing"

# Make sure directory exists
os.makedirs(cache_dir, exist_ok=True)

print(f"Cache exists: {os.path.exists(cache_dir)}")

try:
    _clear_cache([invalid_dataset], cache_dir=cache_dir)
    print("BUG: No ValueError raised!")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")
finally:
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)