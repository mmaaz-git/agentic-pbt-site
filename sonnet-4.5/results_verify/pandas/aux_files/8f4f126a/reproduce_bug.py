import pandas as pd


class TestClass:
    pass


err = pd.errors.AbstractMethodError(TestClass(), methodtype="classmethod")
print(str(err))