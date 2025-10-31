import numpy as np
import scipy.spatial.distance as dist
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays


@given(arrays(np.bool_, (10,), elements=st.booleans()))
def test_dice_identity_property(u):
    d = dist.dice(u, u)
    assert not np.isnan(d), f"dice(u, u) should not be NaN, got {d} for u={u}"


if __name__ == "__main__":
    test_dice_identity_property()