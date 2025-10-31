import pytest
from hypothesis import given, settings, strategies as st
from pandas.io.parsers.readers import validate_integer
import pandas as pd
import io


# First, test the specific failing case mentioned
print("Testing specific failing case: val=-1.0, min_val=0")
try:
    result = validate_integer("test", -1.0, 0)
    print(f"  Result: {result} (NO EXCEPTION RAISED - BUG CONFIRMED)")
except ValueError as e:
    print(f"  ValueError raised: {e}")

print("\nTesting another case: val=-1.0, min_val=1")
try:
    result = validate_integer("chunksize", -1.0, 1)
    print(f"  Result: {result} (NO EXCEPTION RAISED - BUG CONFIRMED)")
except ValueError as e:
    print(f"  ValueError raised: {e}")

print("\nTesting with positive float that meets min_val: val=5.0, min_val=1")
try:
    result = validate_integer("test", 5.0, 1)
    print(f"  Result: {result} (Expected behavior)")
except ValueError as e:
    print(f"  ValueError raised: {e}")

print("\nTesting with integer below min_val: val=-1, min_val=0")
try:
    result = validate_integer("test", -1, 0)
    print(f"  Result: {result} (NO EXCEPTION RAISED)")
except ValueError as e:
    print(f"  ValueError raised as expected: {e}")

print("\nTesting with float that cannot be losslessly converted: val=1.5, min_val=0")
try:
    result = validate_integer("test", 1.5, 0)
    print(f"  Result: {result}")
except ValueError as e:
    print(f"  ValueError raised as expected: {e}")

# Test real usage with pandas read_csv
print("\n\nTesting real usage with pandas.read_csv:")
csv_data = "a,b,c\n1,2,3\n4,5,6"
try:
    reader = pd.read_csv(io.StringIO(csv_data), chunksize=-1.0)
    print(f"  read_csv with chunksize=-1.0 succeeded (BUG CONFIRMED)")
    # Try to read a chunk to see what happens
    try:
        chunk = next(reader)
        print(f"  Successfully read chunk with shape: {chunk.shape}")
    except Exception as e:
        print(f"  Error when trying to read chunk: {e}")
except Exception as e:
    print(f"  Exception raised: {e}")

# Now run the hypothesis test
print("\n\nRunning property-based test:")
failures = []

@settings(max_examples=500)
@given(
    val=st.floats(allow_nan=False, allow_infinity=False),
    min_val=st.integers(min_value=0, max_value=1000)
)
def test_validate_integer_respects_min_val_for_floats(val, min_val):
    if val != int(val):
        return

    if int(val) >= min_val:
        result = validate_integer("test", val, min_val)
        assert result >= min_val
    else:
        with pytest.raises(ValueError, match="must be an integer"):
            validate_integer("test", val, min_val)

# Run the test and catch failures
try:
    test_validate_integer_respects_min_val_for_floats()
    print("Property-based test passed all examples")
except AssertionError as e:
    print(f"Property-based test failed with assertion error")
except Exception as e:
    print(f"Property-based test failed: {e}")