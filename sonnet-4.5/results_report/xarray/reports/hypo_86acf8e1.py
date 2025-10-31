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
            pass  # This is the expected behavior
        except ValueError:
            assert False, "Raised ValueError instead of TypeError"

if __name__ == "__main__":
    test_excess_positional_args_should_raise_typeerror()