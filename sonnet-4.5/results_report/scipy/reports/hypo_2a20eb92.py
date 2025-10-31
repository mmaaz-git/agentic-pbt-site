import scipy.special as sp
from hypothesis import given, strategies as st


@given(st.floats(min_value=1e-10, max_value=1e-02, allow_nan=False, allow_infinity=False))
def test_gammainccinv_inverse_property(a):
    y = 0.5
    x = sp.gammainccinv(a, y)
    result = sp.gammaincc(a, x)
    assert abs(result - y) < 1e-6, \
        f"For a={a}, gammainccinv({a}, {y}) = {x}, but gammaincc({a}, {x}) = {result}, expected {y}"

if __name__ == "__main__":
    test_gammainccinv_inverse_property()