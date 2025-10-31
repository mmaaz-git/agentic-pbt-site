import tempfile
import os
from scipy.datasets._utils import _clear_cache
from scipy.datasets._registry import method_files_map


def fake_dataset():
    pass
fake_dataset.__name__ = "nonexistent_dataset"

print("Testing invalid dataset with non-existent cache directory...")
with tempfile.TemporaryDirectory() as tmpdir:
    nonexistent_cache = os.path.join(tmpdir, "nonexistent")

    try:
        _clear_cache(
            datasets=[fake_dataset],
            cache_dir=nonexistent_cache,
            method_map=method_files_map
        )
        print("No error raised - function returned silently")
    except ValueError as e:
        print(f"ValueError raised as expected: {e}")

print("\n" + "="*50 + "\n")
print("Testing the same invalid dataset with existing cache directory...")
with tempfile.TemporaryDirectory() as tmpdir:
    existing_cache = tmpdir  # tmpdir exists

    try:
        _clear_cache(
            datasets=[fake_dataset],
            cache_dir=existing_cache,
            method_map=method_files_map
        )
        print("No error raised - function returned silently")
    except ValueError as e:
        print(f"ValueError raised as expected: {e}")