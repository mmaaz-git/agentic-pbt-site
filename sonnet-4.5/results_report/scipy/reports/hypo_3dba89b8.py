from hypothesis import given, strategies as st, assume, settings
import numpy as np
from scipy.integrate import cumulative_simpson

@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=3, max_size=100)
)
@settings(max_examples=500)
def test_cumulative_simpson_monotonic_increasing_for_positive(y):
    y_arr = np.array(y)
    assume(np.all(y_arr >= 0))
    assume(np.any(y_arr > 0))

    cumulative_result = cumulative_simpson(y_arr, initial=0)

    diffs = np.diff(cumulative_result)
    assert np.all(diffs >= -1e-10)

if __name__ == "__main__":
    test_cumulative_simpson_monotonic_increasing_for_positive()