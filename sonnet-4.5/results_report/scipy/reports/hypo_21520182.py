from hypothesis import given, settings, example
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
import numpy as np
import scipy.linalg


def invertible_matrices(min_size=2, max_size=5):
    return st.integers(min_value=min_size, max_value=max_size).flatmap(
        lambda n: arrays(
            dtype=np.float64,
            shape=(n, n),
            elements=st.floats(
                min_value=-10,
                max_value=10,
                allow_nan=False,
                allow_infinity=False
            )
        )
    ).filter(lambda A: abs(np.linalg.det(A)) > 1e-10)


@given(invertible_matrices(min_size=2, max_size=4))
@settings(max_examples=50)
@example(np.array([[-1.0, -1e-50], [1.0, -1.0]]))  # Add our specific failing case
def test_logm_expm_round_trip(A):
    logA = scipy.linalg.logm(A)
    if np.any(np.isnan(logA)) or np.any(np.isinf(logA)):
        return
    result = scipy.linalg.expm(logA)
    assert np.allclose(result, A, rtol=1e-4, atol=1e-6), f"Failed for A={A}"

# Run the test and catch the failing example
if __name__ == "__main__":
    try:
        test_logm_expm_round_trip()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")