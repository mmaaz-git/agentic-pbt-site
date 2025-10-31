import pandas.errors


class SampleClass:
    pass


instance = SampleClass()
error = pandas.errors.AbstractMethodError(instance, methodtype="classmethod")

try:
    message = str(error)
    print(f"Message: {message}")
except AttributeError as e:
    print(f"Crashed with AttributeError: {e}")