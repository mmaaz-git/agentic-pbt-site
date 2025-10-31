import pandas.errors

class DummyClass:
    pass

try:
    pandas.errors.AbstractMethodError(DummyClass(), methodtype='foobar')
except ValueError as e:
    print(f"Error message: {e}")