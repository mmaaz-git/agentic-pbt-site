from hypothesis import given, strategies as st
import numpy as np
import warnings

@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5)
)
def test_bmat_string_with_globals(rows, cols):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)

        globals_dict = {
            'A': np.matrix(np.ones((rows, cols))),
            'B': np.matrix(np.zeros((rows, cols)))
        }

        result = np.bmat('A, B', gdict=globals_dict)

        assert result.shape == (rows, 2 * cols)

# Run the test
if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_bmat_string_with_globals()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")