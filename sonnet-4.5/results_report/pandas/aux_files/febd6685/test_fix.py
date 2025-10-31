import pandas.errors as errors

# Test with the documented usage - these should work
class Foo:
    @classmethod
    def classmethod(cls):
        error = errors.AbstractMethodError(cls, methodtype="classmethod")
        print(f"Classmethod with class: {str(error)}")

    def method(self):
        error = errors.AbstractMethodError(self)
        print(f"Method with instance: {str(error)}")

# Test the documented examples work
Foo.classmethod()
Foo().method()

# Test the problematic case - instance with classmethod
class DummyClass:
    pass

instance = DummyClass()
try:
    error = errors.AbstractMethodError(instance, methodtype='classmethod')
    print(f"Instance with classmethod: {str(error)}")
except AttributeError as e:
    print(f"ERROR - Instance with classmethod fails: {e}")

# Test all methodtypes with instance
for methodtype in ["method", "staticmethod", "property"]:
    error = errors.AbstractMethodError(instance, methodtype=methodtype)
    print(f"Instance with {methodtype}: {str(error)}")