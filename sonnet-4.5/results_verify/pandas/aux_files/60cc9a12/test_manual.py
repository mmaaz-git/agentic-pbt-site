import pandas.errors as errors

class DummyClass:
    pass

# Test with an invalid methodtype
try:
    errors.AbstractMethodError(DummyClass(), methodtype='invalid')
except ValueError as e:
    print(f"Error message: {e}")
    print(f"\nExpected: methodtype must be one of {{'method', 'property', 'staticmethod', 'classmethod'}}, got invalid instead.")
    print(f"\nActual message says: 'methodtype must be one of invalid'")

# Test with another invalid methodtype
try:
    errors.AbstractMethodError(DummyClass(), methodtype='custom_method')
except ValueError as e:
    print(f"\nSecond test error message: {e}")