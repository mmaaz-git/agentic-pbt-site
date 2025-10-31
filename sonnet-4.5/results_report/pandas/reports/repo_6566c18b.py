from pandas.io.parsers.readers import validate_integer

# Test case that should fail but doesn't
result = validate_integer("chunksize", -1.0, 1)
print(f"Result: {result}")
print(f"Type: {type(result)}")

# This should raise ValueError but returns -1 instead
print("\nTest passed when it should have raised ValueError!")
print(f"Expected: ValueError('chunksize' must be an integer >=1)")
print(f"Actual: Returned {result}")