import pandas.errors as errors

# Test 1: Using cls in classmethod (as shown in docstring)
class Foo:
    @classmethod
    def classmethod(cls):
        error = errors.AbstractMethodError(cls, methodtype="classmethod")
        print(f"Error with cls: {error}")
        return error

    def method(self):
        error = errors.AbstractMethodError(self)
        print(f"Error with self: {error}")
        return error

print("Testing classmethod with cls:")
try:
    e = Foo.classmethod()
except Exception as ex:
    print(f"Failed: {ex}")

print("\nTesting method with self:")
try:
    e = Foo().method()
except Exception as ex:
    print(f"Failed: {ex}")

# Test 2: Using instance with classmethod (bug scenario)
print("\nTesting instance with methodtype='classmethod':")
instance = Foo()
try:
    error = errors.AbstractMethodError(instance, methodtype='classmethod')
    print(f"Error created: {error}")
except Exception as ex:
    print(f"Failed: {ex}")