from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis import strategies as st
import numpy as np
from pandas.core.dtypes.common import is_numeric_v_string_like


@given(
    arrays(np.int64, shape=st.integers(min_value=1, max_value=10)),
    st.text(min_size=1, max_size=10)
)
@settings(max_examples=500)
def test_is_numeric_v_string_like_symmetric(arr, s):
    result1 = is_numeric_v_string_like(arr, s)
    result2 = is_numeric_v_string_like(s, arr)
    assert result1 == result2, f"Asymmetric for arr={arr}, s={s!r}: {result1} != {result2}"

if __name__ == "__main__":
    test_is_numeric_v_string_like_symmetric()