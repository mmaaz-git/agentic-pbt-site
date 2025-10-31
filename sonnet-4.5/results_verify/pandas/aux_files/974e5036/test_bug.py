import pandas.errors as pd_errors


class DummyClass:
    pass


instance = DummyClass()

print("Testing invalid methodtype 'invalid_type':")
try:
    error = pd_errors.AbstractMethodError(instance, methodtype="invalid_type")
except ValueError as e:
    print(f"Actual error message: {e}")
    print()
    print("Expected error message:")
    print("  methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got 'invalid_type' instead.")

print("\n" + "="*50 + "\n")
print("Testing valid methodtype 'method':")
try:
    error = pd_errors.AbstractMethodError(instance, methodtype="method")
    print(f"Created successfully, message: {error}")
except ValueError as e:
    print(f"Unexpected error: {e}")

print("\n" + "="*50 + "\n")
print("Testing all valid methodtypes:")
valid_types = ["method", "classmethod", "staticmethod", "property"]
for methodtype in valid_types:
    try:
        if methodtype == "classmethod":
            # For classmethod, pass the class itself
            error = pd_errors.AbstractMethodError(DummyClass, methodtype=methodtype)
        else:
            error = pd_errors.AbstractMethodError(instance, methodtype=methodtype)
        print(f"✓ {methodtype}: {error}")
    except ValueError as e:
        print(f"✗ {methodtype}: {e}")