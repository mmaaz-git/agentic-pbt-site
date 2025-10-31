import os
import tempfile
from scipy.datasets._utils import _clear_cache

def invalid_dataset():
    pass

print("Testing bug report...")
print("=" * 50)

with tempfile.TemporaryDirectory() as tmpdir:
    non_existent = os.path.join(tmpdir, "nonexistent")

    print("Test 1: Calling _clear_cache with invalid dataset and non-existent cache dir")
    try:
        _clear_cache(invalid_dataset, cache_dir=non_existent)
        print("Result: No error raised! (This is the reported bug)")
    except ValueError as e:
        print(f"Result: ValueError raised: {e}")
    except Exception as e:
        print(f"Result: Other exception raised: {e}")

    print("\nTest 2: Calling _clear_cache with invalid dataset and existing cache dir")
    try:
        _clear_cache(invalid_dataset, cache_dir=tmpdir)
        print("Result: No error raised!")
    except ValueError as e:
        print(f"Result: ValueError raised: {e}")
    except Exception as e:
        print(f"Result: Other exception raised: {e}")