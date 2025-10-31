from xarray.util.deprecation_helpers import _deprecate_positional_args
import warnings

@_deprecate_positional_args("v0.1.0")
def func(*args, **kwargs):
    return sum(args) + sum(kwargs.values())

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = func(1, 2, a=3)
    print(f"Result: {result}")