import pandas.errors as pd_errors


class DummyClass:
    pass


# Test 1: Pass instance with classmethod (causes the bug)
print("Test 1: Pass instance with methodtype='classmethod'")
dummy_instance = DummyClass()
error1 = pd_errors.AbstractMethodError(dummy_instance, methodtype="classmethod")
try:
    msg1 = str(error1)
    print(f"  Success: {msg1}")
except AttributeError as e:
    print(f"  AttributeError: {e}")

# Test 2: Pass class with classmethod (what the docstring shows)
print("\nTest 2: Pass class with methodtype='classmethod'")
error2 = pd_errors.AbstractMethodError(DummyClass, methodtype="classmethod")
try:
    msg2 = str(error2)
    print(f"  Success: {msg2}")
except AttributeError as e:
    print(f"  AttributeError: {e}")

# Test 3: Pass instance with method (standard case)
print("\nTest 3: Pass instance with methodtype='method'")
error3 = pd_errors.AbstractMethodError(dummy_instance, methodtype="method")
try:
    msg3 = str(error3)
    print(f"  Success: {msg3}")
except AttributeError as e:
    print(f"  AttributeError: {e}")

# Test 4: Check docstring example
print("\nTest 4: Docstring example")
class Foo:
    @classmethod
    def classmethod(cls):
        raise pd_errors.AbstractMethodError(cls, methodtype="classmethod")

    def method(self):
        raise pd_errors.AbstractMethodError(self)

try:
    Foo.classmethod()
except pd_errors.AbstractMethodError as e:
    print(f"  Classmethod error: {e}")

try:
    Foo().method()
except pd_errors.AbstractMethodError as e:
    print(f"  Method error: {e}")