import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.spatial.distance import jensenshannon


@given(
    st.lists(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100)
)
@settings(max_examples=500)
def test_jensenshannon_identity(p_list):
    p = np.array(p_list)
    d = jensenshannon(p, p)
    assert np.isclose(d, 0.0), f"jensenshannon(p, p) should be 0, got {d}"

if __name__ == "__main__":
    test_jensenshannon_identity()