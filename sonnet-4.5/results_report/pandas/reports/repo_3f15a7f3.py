import pandas.errors as pd_errors


class DummyClass:
    pass


dummy = DummyClass()

try:
    pd_errors.AbstractMethodError(dummy, methodtype="invalid_type")
except ValueError as e:
    print(f"Error message: {e}")