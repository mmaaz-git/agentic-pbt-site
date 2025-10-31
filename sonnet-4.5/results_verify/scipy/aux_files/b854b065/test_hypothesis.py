from hypothesis import given, strategies as st, assume
import pytest
import scipy.datasets


@given(st.one_of(st.integers(), st.text(), st.floats()))
def test_clear_cache_non_callable_handling(value):
    assume(not callable(value))

    with pytest.raises((AssertionError, ValueError, TypeError)):
        scipy.datasets.clear_cache(value)

if __name__ == "__main__":
    test_clear_cache_non_callable_handling()