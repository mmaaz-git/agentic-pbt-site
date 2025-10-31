from hypothesis import given, strategies as st
import pytest
import os
import shutil
from scipy.datasets._utils import _clear_cache

@given(st.booleans())
def test_clear_cache_validates_regardless_of_cache_existence(cache_exists):
    def invalid_dataset():
        pass

    invalid_dataset.__name__ = "not_a_real_dataset"

    cache_dir = f"/tmp/test_cache_{cache_exists}"

    if cache_exists:
        os.makedirs(cache_dir, exist_ok=True)
    else:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

    try:
        with pytest.raises(ValueError, match="Dataset method .* doesn't exist"):
            _clear_cache([invalid_dataset], cache_dir=cache_dir)
    finally:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

# Run the test
test_clear_cache_validates_regardless_of_cache_existence()