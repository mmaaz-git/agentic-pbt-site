from hypothesis import given, strategies as st
import numpy as np
from scipy import stats

@given(
    data=st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=100
    )
)
def test_quantile_zero_is_min(data):
    x = np.array(data)
    q0 = stats.quantile(x, 0)
    min_x = np.min(x)
    assert np.allclose(q0, min_x), f"quantile(x, 0) should be min(x)"

if __name__ == "__main__":
    test_quantile_zero_is_min()