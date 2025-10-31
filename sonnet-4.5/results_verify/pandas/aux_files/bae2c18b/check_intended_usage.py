import pandas.errors as errors

# Check what the intended usage is based on the examples in the docstring
# Example 1: classmethod using cls
class TestClass:
    @classmethod
    def test_classmethod(cls):
        # When this is called, cls is the class itself
        print(f"cls type: {type(cls)}")
        print(f"cls has __name__: {hasattr(cls, '__name__')}")
        error = errors.AbstractMethodError(cls, methodtype="classmethod")
        return error

    def test_method(self):
        # When this is called, self is an instance
        print(f"self type: {type(self)}")
        print(f"self has __name__: {hasattr(self, '__name__')}")
        error = errors.AbstractMethodError(self)
        return error

print("=== Testing classmethod ===")
err1 = TestClass.test_classmethod()
print(f"Error message: {err1}")

print("\n=== Testing regular method ===")
err2 = TestClass().test_method()
print(f"Error message: {err2}")

print("\n=== Analysis ===")
print("The docstring examples show:")
print("1. For classmethods: pass cls (the class)")
print("2. For regular methods: pass self (an instance)")
print("\nThe bug occurs when someone passes an instance with methodtype='classmethod'")
print("This is logically inconsistent - if it's a classmethod, the first param should be a class")