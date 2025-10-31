import pandas as pd


class TestClass:
    pass


try:
    pd.errors.AbstractMethodError(TestClass(), methodtype="invalid")
except ValueError as e:
    print(f"Error message: {e}")