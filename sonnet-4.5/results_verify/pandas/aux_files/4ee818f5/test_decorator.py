from pandas.util._decorators import deprecate_nonkeyword_arguments
import warnings

allowed = ["self", "x", "y"]
print(f"Before decoration: {allowed}")

@deprecate_nonkeyword_arguments(version="2.0", allowed_args=allowed)
def my_func(self, x, y, z=1):
    return x + y + z

print(f"After decoration: {allowed}")

with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    result = my_func(None, 1, 2, 3)
    print(f"Function result: {result}")

print(f"After first call: {allowed}")

# Test second call to see if issue persists
with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    result2 = my_func(None, 4, 5, 6)
    print(f"Second function result: {result2}")

print(f"After second call: {allowed}")