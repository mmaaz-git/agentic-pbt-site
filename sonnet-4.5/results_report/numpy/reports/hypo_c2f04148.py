import numpy.rec
import math
from hypothesis import given, strategies as st


@given(st.lists(st.floats(allow_nan=True, allow_infinity=False), min_size=0, max_size=20))
def test_find_duplicate_nan_behavior(lst):
    result = numpy.rec.find_duplicate(lst)
    nan_count = sum(1 for x in lst if isinstance(x, float) and math.isnan(x))

    if nan_count > 1:
        nan_in_result = any(isinstance(x, float) and math.isnan(x) for x in result)
        assert nan_in_result, f"Multiple NaN values in input but none in result"

if __name__ == "__main__":
    test_find_duplicate_nan_behavior()