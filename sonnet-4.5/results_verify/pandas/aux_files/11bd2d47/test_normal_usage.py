import pandas as pd

class Foo:
    @classmethod
    def classmethod(cls):
        raise pd.errors.AbstractMethodError(cls, methodtype="classmethod")

    def method(self):
        raise pd.errors.AbstractMethodError(self)

# Test classmethod (should work)
try:
    test = Foo.classmethod()
except pd.errors.AbstractMethodError as e:
    print(f"Classmethod error: {e}")

# Test instance method (should work)
try:
    test2 = Foo().method()
except pd.errors.AbstractMethodError as e:
    print(f"Instance method error: {e}")