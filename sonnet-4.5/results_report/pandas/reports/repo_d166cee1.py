import pandas.errors


class DummyClass:
    pass


try:
    err = pandas.errors.AbstractMethodError(DummyClass(), methodtype="invalid_type")
except ValueError as e:
    print(f"Error message: {e}")