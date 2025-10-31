import pandas.errors as pd_errors


class DummyClass:
    pass


try:
    pd_errors.AbstractMethodError(DummyClass(), methodtype="foo")
except ValueError as e:
    print(f"Error message: {e}")