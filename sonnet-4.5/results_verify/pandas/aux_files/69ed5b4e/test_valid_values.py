import pandas.errors

# Test valid values
valid_values = ["method", "classmethod", "staticmethod", "property"]

for val in valid_values:
    try:
        err = pandas.errors.AbstractMethodError(object(), methodtype=val)
        print(f"✓ '{val}' is accepted")
    except ValueError as e:
        print(f"✗ '{val}' raised ValueError: {e}")

# Test that it actually works with valid values and produces correct error messages
class TestClass:
    pass

for methodtype in valid_values:
    err = pandas.errors.AbstractMethodError(TestClass(), methodtype=methodtype)
    print(f"Error message for '{methodtype}': {err}")