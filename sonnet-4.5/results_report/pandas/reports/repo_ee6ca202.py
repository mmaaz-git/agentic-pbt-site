import pandas.errors as pd_errors


class DummyClass:
    pass


# This should raise ValueError with swapped variables in error message
pd_errors.AbstractMethodError(DummyClass(), methodtype="foo")