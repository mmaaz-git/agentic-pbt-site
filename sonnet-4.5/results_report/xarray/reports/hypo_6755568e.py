from hypothesis import given, strategies as st
from xarray.util.deprecation_helpers import _deprecate_positional_args
import warnings

@given(st.integers(), st.integers(), st.integers())
def test_decorator_with_varargs(x, y, z):
    @_deprecate_positional_args("v0.1.0")
    def func(*args, **kwargs):
        return sum(args) + sum(kwargs.values())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = func(x, y, a=z)

test_decorator_with_varargs()