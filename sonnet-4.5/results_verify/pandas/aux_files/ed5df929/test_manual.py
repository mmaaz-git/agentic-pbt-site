import pandas.errors

class DummyClass:
    pass

try:
    pandas.errors.AbstractMethodError(DummyClass(), methodtype='foobar')
except ValueError as e:
    print(f"Actual: {e}")
    print(f"Expected: methodtype must be one of {{'method', 'classmethod', 'staticmethod', 'property'}}, got foobar instead.")