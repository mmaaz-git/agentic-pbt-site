import math
from hypothesis import given, settings, strategies as st, assume
import scipy.constants as sc
import warnings

# Suppress constant warnings
warnings.filterwarnings('ignore', category=UserWarning)

all_keys = list(sc.physical_constants.keys())

@given(st.sampled_from(all_keys))
@settings(max_examples=100)  # Reduced for quick test
def test_precision_calculation(key):
    result = sc.precision(key)
    value_const, unit_const, abs_precision = sc.physical_constants[key]

    if value_const == 0:
        print(f"Skipping {key} - zero value")
        assume(False)  # Skip this example
        return

    expected = abs(abs_precision / value_const)

    # Test if they're close using the formula from bug report
    matches = math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-15)

    if not matches:
        # Check if it's a sign issue
        if math.isclose(abs(result), expected, rel_tol=1e-9, abs_tol=1e-15):
            print(f"SIGN ISSUE: {key}")
            print(f"  Value: {value_const}")
            print(f"  Result: {result}")
            print(f"  Expected: {expected}")
            print(f"  Value is negative: {value_const < 0}")
        else:
            print(f"OTHER ISSUE: {key}")
            print(f"  Value: {value_const}")
            print(f"  Result: {result}")
            print(f"  Expected: {expected}")

# Run the test
test_precision_calculation()