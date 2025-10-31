import pytest
from hypothesis import given, strategies as st
from pandas.api.indexers import FixedForwardWindowIndexer


@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=0, max_value=100),
)
def test_step_zero_raises_informative_error(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)

    with pytest.raises((ValueError, ZeroDivisionError)) as exc_info:
        indexer.get_window_bounds(num_values=num_values, step=0)

    if isinstance(exc_info.value, ZeroDivisionError):
        pytest.fail("Should raise ValueError with informative message, not ZeroDivisionError")

# Test with the specific failing input
def test_specific_case():
    indexer = FixedForwardWindowIndexer(window_size=0)
    with pytest.raises((ValueError, ZeroDivisionError)) as exc_info:
        indexer.get_window_bounds(num_values=1, step=0)

    print(f"Exception type: {type(exc_info.value)}")
    print(f"Exception message: {exc_info.value}")

if __name__ == "__main__":
    test_specific_case()