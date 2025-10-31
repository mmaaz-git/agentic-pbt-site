import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
from hypothesis import given, strategies as st, settings


@st.composite
def triangular_system(draw, kind='lower'):
    n = draw(st.integers(min_value=2, max_value=8))
    rows, cols, data = [], [], []

    for i in range(n):
        data.append(draw(st.floats(min_value=0.1, max_value=10)))
        rows.append(i)
        cols.append(i)

    for i in range(n):
        for j in range(i):
            if kind == 'lower' and draw(st.booleans()):
                val = draw(st.floats(min_value=-10, max_value=10))
                rows.append(i)
                cols.append(j)
                data.append(val)

    A = sp.csr_array((data, (rows, cols)), shape=(n, n))
    b = np.array([draw(st.floats(min_value=-100, max_value=100)) for _ in range(n)])
    return A, b


@given(triangular_system(kind='lower'))
@settings(max_examples=10, deadline=None)
def test_spsolve_triangular_lower(system):
    A, b = system
    print(f"Testing with A shape {A.shape}, indices dtype: {A.indices.dtype}")
    try:
        x = linalg.spsolve_triangular(A, b, lower=True)
        result = A @ x
        assert np.allclose(result, b)
        print("  Success (should not happen with int64)")
    except TypeError as e:
        print(f"  TypeError as expected: {e}")
        return  # This is actually expected with int64
    except Exception as e:
        print(f"  Unexpected error: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    print("Running hypothesis test...")
    test_spsolve_triangular_lower()