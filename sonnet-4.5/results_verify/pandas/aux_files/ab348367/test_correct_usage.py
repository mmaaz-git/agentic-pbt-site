import pandas.errors


class SampleClass:
    pass


# Correct usage - passing a class instead of an instance
error = pandas.errors.AbstractMethodError(SampleClass, methodtype="classmethod")

try:
    message = str(error)
    print(f"Message with class: {message}")
except AttributeError as e:
    print(f"Crashed with AttributeError: {e}")