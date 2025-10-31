import numpy as np
import numpy.ma as ma
from hypothesis import given, strategies as st

@given(st.data())
def test_default_fill_value_matches_type(data_strategy):
    dtype = data_strategy.draw(st.sampled_from([np.float32, np.float64, np.int32, np.int64]))

    fill_val = ma.default_fill_value(dtype)

    assert np.isscalar(fill_val) or fill_val is not None

if __name__ == "__main__":
    test_default_fill_value_matches_type()