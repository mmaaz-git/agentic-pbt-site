from hypothesis import given, strategies as st
import pytest
import numpy as np
from pandas import DataFrame
from pandas.core.sample import preprocess_weights

@given(
    n_rows=st.integers(min_value=1, max_value=100),
    axis=st.integers(min_value=0, max_value=1)
)
def test_preprocess_weights_negative_error_message(n_rows, axis):
    df = DataFrame(np.random.randn(n_rows, 3))
    shape = n_rows if axis == 0 else 3
    weights = np.ones(shape, dtype=np.float64)
    weights[0] = -1.0

    with pytest.raises(ValueError) as exc_info:
        preprocess_weights(df, weights, axis)

    error_msg = str(exc_info.value)
    # Bug: message says "many" instead of "may"
    assert "weight vector many not include negative values" == error_msg

if __name__ == "__main__":
    test_preprocess_weights_negative_error_message()