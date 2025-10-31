from hypothesis import given, strategies as st, assume, settings
from scipy.optimize import bisect


@given(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=500)
def test_bisect_funcalls_accurate(a, b):
    assume(abs(a - b) > 1e-8)
    assume(a < b)

    call_count = [0]

    def f(x):
        call_count[0] += 1
        return x - 1.5

    fa, fb = f(a), f(b)
    assume(fa * fb < 0)

    call_count[0] = 0
    root, result = bisect(f, a, b, full_output=True, disp=False)

    actual_calls = call_count[0]
    reported_calls = result.function_calls

    assert actual_calls == reported_calls, \
        f"bisect: reported {reported_calls} calls but actually made {actual_calls}"

if __name__ == "__main__":
    try:
        test_bisect_funcalls_accurate()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")