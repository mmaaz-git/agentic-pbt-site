from xarray.util.deprecation_helpers import _deprecate_positional_args


@_deprecate_positional_args("v1.0.0")
def example_func(a, *, b=1):
    """A simple function with one positional and one keyword-only argument."""
    return a + b


# Try to call with too many positional arguments (3 arguments when it only accepts 1 positional)
result = example_func(1, 2, 3)
print(f"Result: {result}")