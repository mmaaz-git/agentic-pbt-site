import pytest
from hypothesis import given, strategies as st, example
from pandas.api.indexers import FixedForwardWindowIndexer


@given(
    num_values=st.integers(min_value=1, max_value=100),
    window_size=st.integers(min_value=0, max_value=100),
)
@example(num_values=1, window_size=0)  # The specific failing case
def test_step_zero_raises_informative_error(num_values, window_size):
    indexer = FixedForwardWindowIndexer(window_size=window_size)

    with pytest.raises((ValueError, ZeroDivisionError)) as exc_info:
        indexer.get_window_bounds(num_values=num_values, step=0)

    if isinstance(exc_info.value, ZeroDivisionError):
        pytest.fail("Should raise ValueError with informative message, not ZeroDivisionError")