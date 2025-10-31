from hypothesis import given, strategies as st, assume, settings
from scipy.optimize import newton


@given(
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300)
def test_newton_rtol_validation(x0):
    assume(abs(x0) > 0.1)

    def f(x):
        return x**2 - 4

    def fprime(x):
        return 2 * x

    try:
        root = newton(f, x0, fprime=fprime, rtol=-0.1, disp=False)
        assert False, "newton should reject negative rtol"
    except ValueError:
        pass

if __name__ == "__main__":
    test_newton_rtol_validation()
    print("Test completed")