from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.fftpack as fftpack


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100))
@settings(max_examples=100)
def test_hilbert_ihilbert_roundtrip(x_list):
    x = np.array(x_list)
    x = x - np.mean(x)
    if np.abs(np.sum(x)) < 1e-9:
        result = fftpack.hilbert(fftpack.ihilbert(x))
        try:
            assert np.allclose(result, x, atol=1e-9)
            print(f"✓ PASS for array of length {len(x)}")
        except AssertionError:
            print(f"✗ FAIL for array of length {len(x)}")
            print(f"  Input: {x[:5]}{'...' if len(x) > 5 else ''}")
            print(f"  Expected: {x[:5]}{'...' if len(x) > 5 else ''}")
            print(f"  Got: {result[:5]}{'...' if len(result) > 5 else ''}")
            raise

if __name__ == "__main__":
    # Run the test
    print("Running Hypothesis test for Hilbert round-trip property...")
    try:
        test_hilbert_ihilbert_roundtrip()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")