import pandas.errors as pd_errors


class DummyClass:
    pass


dummy = DummyClass()
error = pd_errors.AbstractMethodError(dummy, methodtype="classmethod")

str(error)