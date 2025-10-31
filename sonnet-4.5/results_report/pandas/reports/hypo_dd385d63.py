import numpy as np
from hypothesis import given, strategies as st
from pandas.core.ops.common import _maybe_match_name


class MockObj:
    def __init__(self, name):
        self.name = name


@given(
    values=st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=10)
)
def test_maybe_match_name_equal_arrays(values):
    arr1 = np.array(values)
    arr2 = np.array(values)

    a = MockObj(arr1)
    b = MockObj(arr2)

    result = _maybe_match_name(a, b)

    assert result is not None, f"Expected array {arr1}, got None"
    assert np.array_equal(result, arr1), f"Expected array {arr1}, got {result}"


if __name__ == "__main__":
    test_maybe_match_name_equal_arrays()