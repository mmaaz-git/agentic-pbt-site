from pandas.core.dtypes.common import ensure_python_int, is_integer, is_float, is_scalar
import numpy as np

# Let's trace through the function logic
test_values = [float('inf'), float('-inf'), float('nan')]

for value in test_values:
    print(f"\n=== Testing {value} ===")
    print(f"is_integer({value}) = {is_integer(value)}")
    print(f"is_float({value}) = {is_float(value)}")
    print(f"is_scalar({value}) = {is_scalar(value)}")

    # Check the first condition
    if not (is_integer(value) or is_float(value)):
        print("Would enter first branch (not int or float)")
    else:
        print("Would skip first branch, go to try block")

    # Test what exceptions int() raises
    print(f"\nint({value}) raises:")
    try:
        int(value)
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")

    # Test the assertion
    print(f"\nChecking assertion new_value == value:")
    try:
        new_value = int(value)
        assert new_value == value
    except AssertionError as e:
        print(f"  AssertionError would be caught")
    except Exception as e:
        print(f"  {type(e).__name__} is raised but NOT caught in except clause")

# Let's check if OverflowError is in the caught exceptions
print("\n=== Current exception handling ===")
print("Caught exceptions: TypeError, ValueError, AssertionError")
print("OverflowError is NOT in the list of caught exceptions")