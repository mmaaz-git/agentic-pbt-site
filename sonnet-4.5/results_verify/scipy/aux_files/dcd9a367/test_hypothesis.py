from hypothesis import given, strategies as st
import numpy as np
import scipy.fft


@given(
    st.lists(
        st.complex_numbers(
            allow_nan=False,
            allow_infinity=False,
            min_magnitude=0.0,
            max_magnitude=1e10
        ),
        min_size=1,
        max_size=100
    )
)
def test_rfft_irfft_implicit_n(x):
    x_arr = np.array(x, dtype=complex)
    result = scipy.fft.irfft(x_arr)
    assert result.shape[0] > 0

# Run the test
print("Running Hypothesis property-based test...")
try:
    test_rfft_irfft_implicit_n()
    print("Test passed!")
except Exception as e:
    print(f"Test failed: {e}")
    print("Testing specific failing case x=[0j]:")
    try:
        x_arr = np.array([0j], dtype=complex)
        result = scipy.fft.irfft(x_arr)
        print(f"Result for [0j]: {result}")
    except Exception as e2:
        print(f"Error for [0j]: {e2}")