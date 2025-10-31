from pandas.core.dtypes.common import ensure_python_int
from hypothesis import given, strategies as st, settings
import pytest
import traceback

# First, test the Hypothesis property-based test
print("=== Running Hypothesis Property-Based Test ===")

@given(st.one_of(st.just(float('inf')), st.just(float('-inf')), st.just(float('nan'))))
@settings(max_examples=10)
def test_ensure_python_int_special_floats_raise_typeerror(x):
    with pytest.raises(TypeError):
        ensure_python_int(x)

# Run the hypothesis test
try:
    test_ensure_python_int_special_floats_raise_typeerror()
    print("Hypothesis test passed")
except Exception as e:
    print(f"Hypothesis test failed: {e}")
    traceback.print_exc()

# Now reproduce manually
print("\n=== Manual Reproduction ===")

test_cases = [
    float('inf'),
    float('-inf'),
    float('nan')
]

for test_value in test_cases:
    print(f"\nTesting ensure_python_int({test_value}):")
    try:
        result = ensure_python_int(test_value)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Exception type: {type(e).__name__}")
        print(f"  Exception message: {e}")

# Test what Python's int() does with these values
print("\n=== Python's int() behavior ===")
for test_value in test_cases:
    print(f"\nTesting int({test_value}):")
    try:
        result = int(test_value)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Exception type: {type(e).__name__}")
        print(f"  Exception message: {e}")