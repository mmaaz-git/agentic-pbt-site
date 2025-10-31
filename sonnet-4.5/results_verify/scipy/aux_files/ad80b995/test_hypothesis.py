import os
import tempfile
import pytest
from hypothesis import given, strategies as st, settings
from scipy.datasets._utils import _clear_cache

def make_invalid_dataset(name):
    def invalid():
        pass
    invalid.__name__ = name
    return invalid

@given(st.text(min_size=1).filter(lambda x: x not in ['ascent', 'electrocardiogram', 'face']))
@settings(max_examples=10)
def test_clear_cache_validates_dataset_regardless_of_cache_existence(invalid_name):
    print(f"Testing with invalid_name: {repr(invalid_name)}")
    invalid_dataset = make_invalid_dataset(invalid_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent = os.path.join(tmpdir, "nonexistent")

        # This should raise ValueError but doesn't when cache dir doesn't exist
        try:
            _clear_cache(invalid_dataset, cache_dir=non_existent)
            print(f"  - Non-existent cache: No error raised (BUG)")
        except ValueError as e:
            print(f"  - Non-existent cache: ValueError raised (expected)")

        # This should raise ValueError and does
        try:
            _clear_cache(invalid_dataset, cache_dir=tmpdir)
            print(f"  - Existing cache: No error raised (unexpected)")
        except ValueError as e:
            print(f"  - Existing cache: ValueError raised (expected)")

# Run the test
if __name__ == "__main__":
    test_clear_cache_validates_dataset_regardless_of_cache_existence()