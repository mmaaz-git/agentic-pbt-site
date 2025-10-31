import warnings
from xarray.util.deprecation_helpers import _deprecate_positional_args

def func(x, *, y=0):
    return x + y

# Create decorated version
decorated = _deprecate_positional_args("v0.1.0")(func)

# Try calling with too many positional arguments
print("Calling decorated(1, 2, 3) on func(x, *, y=0)...")
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = decorated(1, 2, 3)
        print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")

print("\n" + "="*50 + "\n")

# Show what happens with undecorated function for comparison
print("For comparison, calling undecorated func(1, 2, 3)...")
try:
    result = func(1, 2, 3)
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")