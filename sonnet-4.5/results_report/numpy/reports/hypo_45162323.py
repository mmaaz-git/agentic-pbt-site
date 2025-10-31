import numpy as np
from hypothesis import assume, given, settings, strategies as st


@given(
    mean=st.floats(allow_nan=False, allow_infinity=False, min_value=1e-10, max_value=1e2),
    scale=st.floats(min_value=1e-10, max_value=1e2),
)
@settings(max_examples=50)
def test_wald_always_positive(mean, scale):
    assume(mean > 0)
    assume(scale > 0)
    rng = np.random.Generator(np.random.PCG64(42))
    result = rng.wald(mean, scale, size=100)
    assert np.all(result > 0), f"wald({mean}, {scale}) produced negative values: min={np.min(result)}"

if __name__ == "__main__":
    test_wald_always_positive()