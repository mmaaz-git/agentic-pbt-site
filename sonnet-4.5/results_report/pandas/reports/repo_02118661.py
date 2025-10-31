import pandas.errors


class DummyClass:
    pass


# Create an instance (not a class)
instance = DummyClass()

# Create AbstractMethodError with methodtype="classmethod" but passing an instance
err = pandas.errors.AbstractMethodError(instance, methodtype="classmethod")

# Try to convert to string - this should crash
try:
    msg = str(err)
    print(f"String representation: {msg}")
except AttributeError as e:
    print(f"AttributeError: {e}")