import numpy as np
from hypothesis import given, settings, strategies as st


@given(shape=st.floats(min_value=1e-10, max_value=1e-3))
@settings(max_examples=100)
def test_gamma_always_positive(shape):
    rng = np.random.Generator(np.random.PCG64(42))
    result = rng.gamma(shape, size=100)
    assert np.all(result > 0), f"gamma({shape}) produced zeros or negative values: min={np.min(result)}"

if __name__ == "__main__":
    test_gamma_always_positive()