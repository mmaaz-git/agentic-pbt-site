import numpy as np
from hypothesis import given, strategies as st


@given(st.integers(2, 10), st.integers(2, 10))
def test_matrix_remains_2d_after_operations(rows, cols):
    m = np.matrix(np.random.randn(rows, cols))
    assert m.ndim == 2

    result = m[:, np.newaxis, :]
    assert result.ndim == 2, f"Expected 2D matrix, got {result.ndim}D with shape {result.shape}"


if __name__ == "__main__":
    test_matrix_remains_2d_after_operations()