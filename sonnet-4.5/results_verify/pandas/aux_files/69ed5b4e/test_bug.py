import pandas.errors

# Test the simple reproduction case
try:
    pandas.errors.AbstractMethodError(object(), methodtype="invalid")
except ValueError as e:
    print(f"Actual:   {e}")
    print(f"Expected: methodtype must be one of {{'method', 'classmethod', 'staticmethod', 'property'}}, got invalid instead.")