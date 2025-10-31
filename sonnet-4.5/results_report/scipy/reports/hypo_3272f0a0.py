import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages')

import numpy as np
from scipy.spatial import geometric_slerp
from hypothesis import given, strategies as st, settings


def unit_vector_strategy(dims):
    def make_unit_vector(coeffs):
        v = np.array(coeffs)
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            v = np.zeros(len(coeffs))
            v[0] = 1.0
            return v
        return v / norm

    return st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3),
        min_size=dims,
        max_size=dims
    ).map(make_unit_vector)


@settings(max_examples=100)
@given(
    unit_vector_strategy(3),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
)
def test_geometric_slerp_scalar_shape_consistency(start, t):
    end_same = start.copy()
    end_different = -start

    result_same = geometric_slerp(start, end_same, t)
    result_different = geometric_slerp(start, end_different, t)

    assert result_same.shape == result_different.shape, \
        f"Shape mismatch: {result_same.shape} vs {result_different.shape}"


if __name__ == "__main__":
    test_geometric_slerp_scalar_shape_consistency()