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

# For 'method', 'staticmethod', 'property' - use instance
for methodtype in ["method", "staticmethod", "property"]:
    err = pandas.errors.AbstractMethodError(TestClass(), methodtype=methodtype)
    print(f"Error message for '{methodtype}': {err}")

# For 'classmethod' - use class itself
err = pandas.errors.AbstractMethodError(TestClass, methodtype="classmethod")
print(f"Error message for 'classmethod': {err}")