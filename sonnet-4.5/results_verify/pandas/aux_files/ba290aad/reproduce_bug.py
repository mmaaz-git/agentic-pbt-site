import pandas.errors as errors


class DummyClass:
    pass


instance = DummyClass()

try:
    errors.AbstractMethodError(instance, methodtype="invalid")
except ValueError as e:
    print(f"Actual: {e}")

print("Expected: methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got invalid instead.")