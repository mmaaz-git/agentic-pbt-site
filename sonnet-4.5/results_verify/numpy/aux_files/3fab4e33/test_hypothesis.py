import numpy as np
from hypothesis import given, strategies as st


@given(st.integers(2, 10), st.integers(2, 10))
def test_matrix_remains_2d_after_operations(rows, cols):
    m = np.matrix(np.random.randn(rows, cols))
    assert m.ndim == 2

    result = m[:, np.newaxis, :]
    assert result.ndim == 2, f"Expected 2D matrix, got {result.ndim}D with shape {result.shape}"

# Run the test
if __name__ == "__main__":
    print("Testing with Hypothesis...")
    try:
        test_matrix_remains_2d_after_operations()
        print("All tests passed")
    except AssertionError as e:
        print(f"Test failed as expected: {e}")
        print("✓ Hypothesis test confirms the bug")