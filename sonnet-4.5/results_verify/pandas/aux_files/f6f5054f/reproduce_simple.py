import pandas.errors

try:
    pandas.errors.AbstractMethodError(object(), methodtype="invalid")
except ValueError as e:
    print(f"Actual error message: {e}")
    print(f"\nExpected error message: methodtype must be one of {{'method', 'classmethod', 'staticmethod', 'property'}}, got invalid instead.")