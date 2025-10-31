import pandas as pd


class DummyClass:
    pass


instance = DummyClass()

try:
    pd.errors.AbstractMethodError(instance, methodtype="invalid_type")
except ValueError as e:
    print(f"Error message: {e}")