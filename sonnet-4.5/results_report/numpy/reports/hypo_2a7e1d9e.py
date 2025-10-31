import numpy as np
from hypothesis import given, strategies as st

@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5)
)
def test_bmat_gdict_without_ldict(rows, cols):
    """Test that bmat works with gdict provided but ldict=None.

    According to documentation, both ldict and gdict are optional parameters.
    This test verifies that providing only gdict (with ldict=None) should work.
    """
    A = np.matrix(np.ones((rows, cols)))
    result = np.bmat('A', ldict=None, gdict={'A': A})
    assert np.array_equal(result, A), f"Expected result to equal A, got {result}"

if __name__ == "__main__":
    # Run the property-based test
    test_bmat_gdict_without_ldict()