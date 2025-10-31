import pandas.errors as errors


class DummyClass:
    pass


instance = DummyClass()

try:
    errors.AbstractMethodError(instance, methodtype="invalid")
except ValueError as e:
    print(f"Actual error message: {e}")
    print()
    print("Expected error message: methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got invalid instead.")