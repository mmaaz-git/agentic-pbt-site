import pandas.errors

try:
    pandas.errors.AbstractMethodError(object(), methodtype="invalid")
except ValueError as e:
    print(f"Error message: {e}")