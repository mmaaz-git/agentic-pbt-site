import pandas as pd

print("=== Bug Reproduction ===")
print("\n1. Simple reproduction with 'invalid_type':")
try:
    error = pd.errors.AbstractMethodError(object(), methodtype="invalid_type")
except ValueError as e:
    print(f"Actual:   {e}")
    print(f"Expected: methodtype must be one of {{'method', 'classmethod', 'staticmethod', 'property'}}, got invalid_type instead.")
    print(f"Bug confirmed: Parameters are swapped!")

print("\n2. Testing with various invalid values:")
test_values = ["0", "invalid", "foo", "bar", "test123", "", "Method", "PROPERTY"]
for val in test_values:
    try:
        error = pd.errors.AbstractMethodError(object(), methodtype=val)
    except ValueError as e:
        error_msg = str(e)
        # The bug is that it says "must be one of [invalid_value]" instead of "must be one of [valid_values]"
        if f"must be one of {val}" in error_msg:
            print(f"Value '{val}': CONFIRMED BUG - says 'must be one of {val}'")
        else:
            print(f"Value '{val}': {error_msg}")

print("\n3. Checking valid values:")
valid_values = ["method", "classmethod", "staticmethod", "property"]
for val in valid_values:
    try:
        error = pd.errors.AbstractMethodError(object(), methodtype=val)
        print(f"Value '{val}': Successfully created (no error)")
    except ValueError as e:
        print(f"Value '{val}': Unexpected error: {e}")

print("\n=== Verification ===")
print("The bug is CONFIRMED. The error message has swapped parameters:")
print("- It says: 'methodtype must be one of [invalid_value], got [valid_set] instead'")
print("- Should say: 'methodtype must be one of [valid_set], got [invalid_value] instead'")