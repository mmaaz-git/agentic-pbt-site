import pandas.errors as errors


class TestClass:
    pass


try:
    errors.AbstractMethodError(TestClass, methodtype="invalid_type")
except ValueError as e:
    print(f"Actual: {e}")
    print()
    print("Expected: methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got 'invalid_type' instead.")