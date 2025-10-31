import tempfile
import os
import pytest
from hypothesis import given, strategies as st
from scipy.datasets._utils import _clear_cache
from scipy.datasets._registry import method_files_map


@given(st.booleans())
def test_clear_cache_validates_dataset_before_checking_cache_exists(cache_exists):
    """
    Property: _clear_cache should validate dataset parameters consistently,
    regardless of whether the cache directory exists or not.

    This test checks that invalid datasets raise ValueError both when
    the cache directory exists and when it doesn't exist.
    """
    def fake_dataset():
        """A dataset function that doesn't exist in the registry"""
        pass
    fake_dataset.__name__ = "nonexistent_dataset"

    with tempfile.TemporaryDirectory() as tmpdir:
        if cache_exists:
            # Use the existing temp directory
            cache_dir = tmpdir
        else:
            # Use a non-existent subdirectory
            cache_dir = os.path.join(tmpdir, "nonexistent_cache_dir")

        print(f"\nTest with cache_exists={cache_exists}")
        print(f"Cache directory: {cache_dir}")
        print(f"Actual cache exists: {os.path.exists(cache_dir)}")

        # Both cases should raise ValueError for invalid dataset
        with pytest.raises(ValueError, match="Dataset method .* doesn't exist"):
            _clear_cache(
                datasets=[fake_dataset],
                cache_dir=cache_dir,
                method_map=method_files_map
            )
        print(f"✓ ValueError correctly raised when cache_exists={cache_exists}")


if __name__ == "__main__":
    # Run the property-based test
    test_clear_cache_validates_dataset_before_checking_cache_exists()