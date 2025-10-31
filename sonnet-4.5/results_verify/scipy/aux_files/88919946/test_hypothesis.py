#!/usr/bin/env python3
"""Run the property-based test from the bug report"""

import numpy as np
import scipy.fftpack as fftpack
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis import strategies as st

@given(
    arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    ),
    st.integers(min_value=1, max_value=4)
)
def test_dct_handles_all_sizes(x, dct_type):
    result = fftpack.dct(x, type=dct_type, norm='ortho')
    assert len(result) == len(x)

# Run the test
if __name__ == "__main__":
    print("Running property-based test...")
    try:
        test_dct_handles_all_sizes()
        print("Test passed for all generated cases!")
    except Exception as e:
        print(f"Test failed with error: {e}")

    # Test the specific failing case
    print("\nTesting specific failing case from bug report:")
    print("x = np.array([1.]), dct_type=1")
    try:
        x = np.array([1.])
        result = fftpack.dct(x, type=1, norm='ortho')
        print(f"Result: {result}")
        print(f"Length check: input={len(x)}, output={len(result)}")
    except Exception as e:
        print(f"Failed as expected: {type(e).__name__}: {e}")