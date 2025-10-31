import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st, settings
from xarray.core.nputils import inverse_permutation


@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=50, unique=True))
@settings(max_examples=1000)
def test_inverse_permutation_involution(indices_list):
    """Test that applying inverse_permutation twice returns to the original for partial permutations."""
    indices = np.array(indices_list, dtype=np.intp)
    N = np.max(indices) + 1

    inv1 = inverse_permutation(indices, N)
    inv2 = inverse_permutation(inv1, N)

    assert np.array_equal(inv2, indices), f"Double inverse should return original: {indices} -> {inv1} -> {inv2}"

if __name__ == "__main__":
    test_inverse_permutation_involution()