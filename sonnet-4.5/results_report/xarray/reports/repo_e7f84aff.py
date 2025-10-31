from xarray.util.deprecation_helpers import _deprecate_positional_args


@_deprecate_positional_args("v0.1.0")
def example_func(a, *, b=2):
    return a + b


# This should raise TypeError but crashes with ValueError instead
example_func(1, 2, 3)