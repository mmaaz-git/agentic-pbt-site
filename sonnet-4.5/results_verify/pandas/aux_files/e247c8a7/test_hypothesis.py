import numpy as np
from hypothesis import given, strategies as st, example
from pandas.core.ops.common import _maybe_match_name


class MockObj:
    def __init__(self, name):
        self.name = name


@given(
    values=st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=10)
)
@example([0, 0])  # Specific failing case
def test_maybe_match_name_equal_arrays(values):
    arr1 = np.array(values)
    arr2 = np.array(values)

    a = MockObj(arr1)
    b = MockObj(arr2)

    result = _maybe_match_name(a, b)

    assert result is not None, f"Expected array {arr1}, but got None"
    assert np.array_equal(result, arr1), f"Expected {arr1}, but got {result}"

if __name__ == "__main__":
    # Run the test
    test_maybe_match_name_equal_arrays()