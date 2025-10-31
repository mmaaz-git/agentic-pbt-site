import pandas.errors as pe


class DummyClass:
    pass


obj = DummyClass()

try:
    pe.AbstractMethodError(obj, methodtype='0')
except ValueError as e:
    print(f"Actual error message:")
    print(f"  {e}")
    print()
    print(f"Expected error message:")
    print(f"  methodtype must be one of {{'method', 'classmethod', 'staticmethod', 'property'}}, got 0 instead.")