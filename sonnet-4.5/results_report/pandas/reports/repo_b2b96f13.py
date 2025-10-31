import pandas as pd


class Foo:
    pass


try:
    pd.errors.AbstractMethodError(Foo(), methodtype="invalid_type")
except ValueError as e:
    print(str(e))