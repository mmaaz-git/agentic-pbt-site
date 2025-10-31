import pandas.errors as pd_errors


class DummyClass:
    pass


instance = DummyClass()

try:
    pd_errors.AbstractMethodError(instance, methodtype="invalid")
except ValueError as e:
    print(f"Actual:   {e}")
    print("Expected: methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got invalid instead.")