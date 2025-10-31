import pandas.errors as pd_errors


class DummyClass:
    pass


instance = DummyClass()

error = pd_errors.AbstractMethodError(instance, methodtype="classmethod")

print("Error created successfully")

try:
    error_str = str(error)
    print(f"Error string: {error_str}")
except AttributeError as e:
    print(f"CRASH: {e}")