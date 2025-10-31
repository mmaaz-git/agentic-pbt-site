from hypothesis import given, strategies as st
import warnings
from xarray.util.deprecation_helpers import _deprecate_positional_args

@given(st.integers(), st.integers(), st.integers())
def test_excess_positional_args_should_raise_typeerror(x, y, z):
    def func(a, *, b=0):
        return a + b

    decorated = _deprecate_positional_args("v0.1.0")(func)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            decorated(x, y, z)
            assert False, "Should have raised an exception"
        except TypeError:
            pass  # This is expected
        except ValueError as e:
            assert False, f"Raised ValueError instead of TypeError: {e}"

# Run the test
print("Running Hypothesis test...")
try:
    test_excess_positional_args_should_raise_typeerror()
    print("Test failed - no error detected in any examples")
except AssertionError as e:
    print(f"Test failure confirmed: {e}")

# Test with specific example
print("\nTesting with specific example (1, 2, 3):")
try:
    def func(a, *, b=0):
        return a + b

    decorated = _deprecate_positional_args("v0.1.0")(func)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        decorated(1, 2, 3)
        print("No error raised")
except TypeError as e:
    print(f"TypeError raised (expected): {e}")
except ValueError as e:
    print(f"ValueError raised (bug confirmed): {e}")