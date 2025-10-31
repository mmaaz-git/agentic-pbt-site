from hypothesis import given, strategies as st, settings
import numpy as np
from scipy.integrate import simpson

@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=3, max_size=100)
)
@settings(max_examples=500)
def test_simpson_reverse_negates(y):
    n = len(y)
    y_arr = np.array(y)
    x_sorted = np.arange(n, dtype=float)

    result_forward = simpson(y_arr, x=x_sorted)
    result_backward = simpson(y_arr[::-1], x=x_sorted[::-1])

    assert np.isclose(result_forward, -result_backward, rtol=1e-10, atol=1e-10), \
        f"Forward: {result_forward}, Backward: {result_backward}, Expected: {-result_forward}"

# Test with the specific failing input
def test_specific_case():
    y = [0.0, 0.0, 0.0, 1.0]
    n = len(y)
    y_arr = np.array(y)
    x_sorted = np.arange(n, dtype=float)

    result_forward = simpson(y_arr, x=x_sorted)
    result_backward = simpson(y_arr[::-1], x=x_sorted[::-1])

    print(f"Specific case - Forward: {result_forward}, Backward: {result_backward}")
    assert np.isclose(result_forward, -result_backward, rtol=1e-10, atol=1e-10), \
        f"Forward: {result_forward}, Backward: {result_backward}, Expected: {-result_forward}"

if __name__ == "__main__":
    print("Testing specific case...")
    try:
        test_specific_case()
        print("Specific case passed!")
    except AssertionError as e:
        print(f"Specific case FAILED: {e}")

    print("\nRunning hypothesis tests...")
    test_simpson_reverse_negates()
    print("All tests passed!")