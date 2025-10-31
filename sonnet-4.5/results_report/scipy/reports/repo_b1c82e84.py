import tempfile
import os
from scipy.datasets._utils import _clear_cache
from scipy.datasets._registry import method_files_map


def fake_dataset():
    """A fake dataset function that is not in the registry"""
    pass

fake_dataset.__name__ = "nonexistent_dataset"

print("Test Case 1: Invalid dataset with non-existent cache directory")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    nonexistent_cache = os.path.join(tmpdir, "nonexistent")

    print(f"Cache directory: {nonexistent_cache}")
    print(f"Cache exists: {os.path.exists(nonexistent_cache)}")
    print("\nCalling _clear_cache with invalid dataset...")

    try:
        _clear_cache(
            datasets=[fake_dataset],
            cache_dir=nonexistent_cache,
            method_map=method_files_map
        )
        print("\n[ERROR] No exception raised! Function returned silently.")
    except ValueError as e:
        print(f"\n[EXPECTED] ValueError raised: {e}")
    except Exception as e:
        print(f"\n[UNEXPECTED] Exception raised: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Test Case 2: Invalid dataset with existing cache directory")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    existing_cache = tmpdir

    print(f"Cache directory: {existing_cache}")
    print(f"Cache exists: {os.path.exists(existing_cache)}")
    print("\nCalling _clear_cache with invalid dataset...")

    try:
        _clear_cache(
            datasets=[fake_dataset],
            cache_dir=existing_cache,
            method_map=method_files_map
        )
        print("\n[ERROR] No exception raised! Function returned silently.")
    except ValueError as e:
        print(f"\n[EXPECTED] ValueError raised: {e}")
    except Exception as e:
        print(f"\n[UNEXPECTED] Exception raised: {type(e).__name__}: {e}")