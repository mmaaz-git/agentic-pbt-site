from hypothesis import given, strategies as st
import numpy as np
import scipy.fft


@given(
    st.lists(
        st.floats(
            allow_nan=False,
            allow_infinity=False,
            min_value=-1e10,
            max_value=1e10
        ),
        min_size=1,
        max_size=1000
    )
)
def test_rfft_irfft_round_trip(data):
    x = np.array(data)
    result = scipy.fft.irfft(scipy.fft.rfft(x))
    assert np.allclose(result, x, rtol=1e-10, atol=1e-12)

if __name__ == "__main__":
    # Test with the reported failing input directly (not through Hypothesis)
    print("Testing with single element array [0.0]...")
    x = np.array([0.0])
    try:
        result = scipy.fft.irfft(scipy.fft.rfft(x))
        print(f"Result: {result}")
        if np.allclose(result, x, rtol=1e-10, atol=1e-12):
            print("Test passed!")
        else:
            print("Test failed - result doesn't match original")
    except Exception as e:
        print(f"Test failed with exception: {e}")