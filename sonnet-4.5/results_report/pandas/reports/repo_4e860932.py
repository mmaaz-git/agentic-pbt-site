import pandas.errors as errors


class TestClass:
    pass


try:
    errors.AbstractMethodError(TestClass, methodtype="invalid_type")
except ValueError as e:
    print(f"Error message: {e}")