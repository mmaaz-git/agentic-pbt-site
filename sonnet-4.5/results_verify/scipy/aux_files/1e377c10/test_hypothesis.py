import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st, settings
from xarray.core.nputils import inverse_permutation


@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=50, unique=True))
@settings(max_examples=1000)
def test_inverse_permutation_involution(indices_list):
    indices = np.array(indices_list, dtype=np.intp)
    N = np.max(indices) + 1

    inv1 = inverse_permutation(indices, N)
    inv2 = inverse_permutation(inv1, N)

    assert np.array_equal(inv2, indices), f"Double inverse should return original: {indices} -> {inv1} -> {inv2}"

# Run the test with hypothesis
if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_inverse_permutation_involution()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")