from pandas.io.parsers.readers import validate_integer

# Test case demonstrating the inconsistent behavior
try:
    result = validate_integer("test_param", -5, min_val=0)
    print(f"Integer -5: {result}")
except ValueError as e:
    print(f"Integer -5: ValueError - {e}")

try:
    result = validate_integer("test_param", -5.0, min_val=0)
    print(f"Float -5.0: {result}")
except ValueError as e:
    print(f"Float -5.0: ValueError - {e}")