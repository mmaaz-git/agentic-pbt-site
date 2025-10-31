import pandas.errors as pe


class DummyClass:
    pass


print("Testing the basic reproduction case:")
try:
    pe.AbstractMethodError(DummyClass(), methodtype="invalid_type")
except ValueError as e:
    print(f"Actual:   {e}")
    print("Expected: methodtype must be one of {{'method', 'classmethod', 'staticmethod', 'property'}}, got invalid_type instead.")

print("\nTesting with another invalid methodtype:")
try:
    pe.AbstractMethodError(DummyClass(), methodtype="0")
except ValueError as e:
    print(f"Actual:   {e}")
    print("Expected: methodtype must be one of {{'method', 'classmethod', 'staticmethod', 'property'}}, got 0 instead.")