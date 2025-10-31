import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
import numpy.ma as ma
import numpy as np


@given(
    m1=st.lists(st.booleans(), min_size=1, max_size=50),
    m2=st.lists(st.booleans(), min_size=1, max_size=50)
)
def test_mask_or_symmetry(m1, m2):
    assume(len(m1) == len(m2))

    result1 = ma.mask_or(m1, m2)
    result2 = ma.mask_or(m2, m1)

    if result1 is ma.nomask and result2 is ma.nomask:
        pass
    elif result1 is ma.nomask or result2 is ma.nomask:
        assert False, f"mask_or should be symmetric, but one is nomask: {result1} vs {result2}"
    else:
        assert np.array_equal(result1, result2), f"mask_or not symmetric: {result1} vs {result2}"

if __name__ == "__main__":
    test_mask_or_symmetry()