import pandas.errors as pd_errors


class DummyClass:
    pass


instance = DummyClass()

try:
    error = pd_errors.AbstractMethodError(instance, methodtype="invalid_type")
except ValueError as e:
    print(f"Actual error message: {e}")
    print()
    print("Expected error message:")
    print("  methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got 'invalid_type' instead.")