import pandas as pd


class Foo:
    pass


try:
    pd.errors.AbstractMethodError(Foo(), methodtype="invalid_type")
except ValueError as e:
    print(f"Actual output:\n{str(e)}")
    print(f"\nExpected output:\nmethodtype must be one of {{'staticmethod', 'method', 'property', 'classmethod'}}, got invalid_type instead.")