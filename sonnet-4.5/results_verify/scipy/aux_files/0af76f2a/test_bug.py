import numpy as np
import scipy.signal.windows as w
from hypothesis import given, strategies as st, settings, assume

# First, run the hypothesis test
@given(
    M=st.integers(min_value=4, max_value=200),
    nbar=st.integers(min_value=1, max_value=20),
    sll=st.floats(min_value=-100.0, max_value=-10.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=10)
def test_taylor_symmetry(M, nbar, sll):
    assume(nbar < M)
    window = w.taylor(M, nbar=nbar, sll=sll, sym=True)
    assert np.allclose(window, window[::-1], rtol=1e-10)

print("Running hypothesis test with negative sll values...")
try:
    test_taylor_symmetry()
    print("Test passed")
except Exception as e:
    print(f"Test failed: {e}")

# Now run the specific reproduction case
print("\n" + "="*50)
print("Specific reproduction case from bug report:")
print("="*50)

window = w.taylor(4, nbar=2, sll=-10.0, sym=True)
print(f"Result with sll=-10.0: {window}")
print(f"Contains NaN: {np.any(np.isnan(window))}")

window_positive = w.taylor(4, nbar=2, sll=10.0, sym=True)
print(f"\nWith positive sll=10.0: {window_positive}")
print(f"Contains NaN: {np.any(np.isnan(window_positive))}")

# Let's also test boundary cases
print("\n" + "="*50)
print("Testing boundary cases:")
print("="*50)

# Test with sll=0
print("\nTesting with sll=0:")
try:
    window_zero = w.taylor(4, nbar=2, sll=0.0, sym=True)
    print(f"Result with sll=0.0: {window_zero}")
    print(f"Contains NaN: {np.any(np.isnan(window_zero))}")
    print(f"Contains Inf: {np.any(np.isinf(window_zero))}")
except Exception as e:
    print(f"Exception raised: {e}")

# Test with very small positive sll
print("\nTesting with sll=0.001:")
try:
    window_small = w.taylor(4, nbar=2, sll=0.001, sym=True)
    print(f"Result with sll=0.001: {window_small}")
    print(f"Contains NaN: {np.any(np.isnan(window_small))}")
except Exception as e:
    print(f"Exception raised: {e}")

# Test with very large negative sll
print("\nTesting with sll=-1000:")
try:
    window_large_neg = w.taylor(4, nbar=2, sll=-1000.0, sym=True)
    print(f"Result with sll=-1000.0: {window_large_neg}")
    print(f"Contains NaN: {np.any(np.isnan(window_large_neg))}")
except Exception as e:
    print(f"Exception raised: {e}")