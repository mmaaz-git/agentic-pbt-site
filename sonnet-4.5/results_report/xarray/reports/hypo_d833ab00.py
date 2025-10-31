from hypothesis import given, strategies as st
from xarray.util.deprecation_helpers import _deprecate_positional_args


@given(st.integers(min_value=2, max_value=5))
def test_excess_positional_args_should_not_crash(n_extra):
    """Test that the decorator handles excess positional arguments gracefully.

    When a function is called with more positional arguments than it can accept,
    Python should raise a TypeError. The decorator should not interfere with this
    by raising its own ValueError about zip lengths.
    """
    @_deprecate_positional_args("v1.0.0")
    def func(a, *, b=1):
        return a + b

    # Create arguments: 1 valid positional + n_extra excess arguments
    args = tuple(range(1 + n_extra))

    try:
        result = func(*args)
        # If we get here, the function accepted the arguments (shouldn't happen)
        print(f"Unexpected success with {len(args)} args: {result}")
    except TypeError as e:
        # This is the expected behavior - Python raises TypeError for too many args
        print(f"Good: Got expected TypeError with {len(args)} args: {e}")
    except ValueError as e:
        if "zip()" in str(e):
            # This is the bug - internal ValueError about zip
            raise AssertionError(f"BUG: Decorator raised internal ValueError about zip with {len(args)} args: {e}")
        else:
            # Some other ValueError (unexpected)
            print(f"Unexpected ValueError with {len(args)} args: {e}")


if __name__ == "__main__":
    # Run the test with different numbers of excess arguments
    test_excess_positional_args_should_not_crash()
    print("\nTest completed!")