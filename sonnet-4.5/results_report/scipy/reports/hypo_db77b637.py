import math

import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as npst
from scipy.spatial import distance


@given(
    u=npst.arrays(
        dtype=np.bool_,
        shape=st.integers(min_value=1, max_value=100),
    ),
    v=npst.arrays(
        dtype=np.bool_,
        shape=st.integers(min_value=1, max_value=100),
    ),
)
@settings(max_examples=300)
def test_dice_symmetry(u, v):
    if u.shape != v.shape:
        return

    d_uv = distance.dice(u, v)
    d_vu = distance.dice(v, u)

    assert math.isclose(d_uv, d_vu, rel_tol=1e-9, abs_tol=1e-9)


if __name__ == "__main__":
    test_dice_symmetry()