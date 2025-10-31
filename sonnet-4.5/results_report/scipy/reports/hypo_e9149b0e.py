from hypothesis import given, strategies as st, assume
from scipy.optimize import bisect, ridder, brenth, brentq


@given(
    root=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    offset=st.floats(min_value=0.1, max_value=50, allow_nan=False, allow_infinity=False),
)
def test_iterations_field_boundary_root(root, offset):
    assume(abs(root) < 100)
    assume(offset > 0.1)

    def f(x):
        return x - root

    a = root - offset
    b = root

    for method in [bisect, ridder, brenth, brentq]:
        root_val, info = method(f, a, b, full_output=True)

        assert isinstance(info.iterations, int)
        assert 0 <= info.iterations <= 1000, \
            f"{method.__name__}: iterations = {info.iterations} (should be small non-negative int)"


if __name__ == "__main__":
    test_iterations_field_boundary_root()