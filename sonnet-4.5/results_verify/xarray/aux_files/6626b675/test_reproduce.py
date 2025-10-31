import warnings
from xarray.util.deprecation_helpers import _deprecate_positional_args

print("Test 1: Reproducing the basic bug")
print("-" * 50)

def func(x, *, y=0):
    return x + y

decorated = _deprecate_positional_args("v0.1.0")(func)

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = decorated(1, 2, 3)
        print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")
except TypeError as e:
    print(f"TypeError raised: {e}")

print("\nTest 2: Property-based test case")
print("-" * 50)

def test_func(a, *, b=0):
    return a + b

decorated_test = _deprecate_positional_args("v0.1.0")(test_func)

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = decorated_test(1, 2, 3)
        print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")
except TypeError as e:
    print(f"TypeError raised: {e}")

print("\nTest 3: Direct function call (without decorator)")
print("-" * 50)

try:
    result = func(1, 2, 3)
    print(f"Result: {result}")
except TypeError as e:
    print(f"TypeError raised (expected): {e}")