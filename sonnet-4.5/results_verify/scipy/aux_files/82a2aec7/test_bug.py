import tempfile
import os
import pytest
from scipy.datasets._utils import _clear_cache
from scipy.datasets._registry import method_files_map


def test_clear_cache_validates_dataset_before_checking_cache_exists():
    def fake_dataset():
        pass
    fake_dataset.__name__ = "nonexistent_dataset"

    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent_cache = os.path.join(tmpdir, "nonexistent_cache_dir")

        with pytest.raises(ValueError, match="doesn't exist"):
            _clear_cache(
                datasets=[fake_dataset],
                cache_dir=nonexistent_cache,
                method_map=method_files_map
            )

if __name__ == "__main__":
    test_clear_cache_validates_dataset_before_checking_cache_exists()
    print("Test passed!")