"""Test to reproduce the attrs or_ validator bug"""
import attr
from attr import validators


# First, let's reproduce the basic bug case
class BuggyValidator:
    def __call__(self, inst, attr, value):
        raise AttributeError("Bug in validator implementation!")


combined = validators.or_(BuggyValidator(), validators.instance_of(str))

@attr.define
class TestClass:
    x: int = attr.field(validator=combined)

print("Testing basic reproduction case:")
try:
    TestClass(x=42)
    print("No exception raised - unexpected!")
except AttributeError as e:
    print(f"GOOD: AttributeError propagated: {e}")
except ValueError as e:
    print(f"BUG: or_ hid the AttributeError, raised ValueError instead: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Let's also test with other exception types
class KeyErrorValidator:
    def __call__(self, inst, attr, value):
        raise KeyError("This is a KeyError!")

combined2 = validators.or_(KeyErrorValidator(), validators.instance_of(str))

@attr.define
class TestClass2:
    x: int = attr.field(validator=combined2)

print("Testing with KeyError:")
try:
    TestClass2(x=42)
    print("No exception raised - unexpected!")
except KeyError as e:
    print(f"GOOD: KeyError propagated: {e}")
except ValueError as e:
    print(f"BUG: or_ hid the KeyError, raised ValueError instead: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test with a validator that raises ValueError (expected to be caught)
class ValueErrorValidator:
    def __call__(self, inst, attr, value):
        raise ValueError("This is a ValueError!")

combined3 = validators.or_(ValueErrorValidator(), validators.instance_of(str))

@attr.define
class TestClass3:
    x: str = attr.field(validator=combined3)

print("Testing with ValueError (should be caught and move to next validator):")
try:
    TestClass3(x="valid string")
    print("SUCCESS: String value accepted by second validator")
except ValueError as e:
    print(f"ValueError raised: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test with TypeError (also expected to be caught according to bug report)
class TypeErrorValidator:
    def __call__(self, inst, attr, value):
        raise TypeError("This is a TypeError!")

combined4 = validators.or_(TypeErrorValidator(), validators.instance_of(str))

@attr.define
class TestClass4:
    x: str = attr.field(validator=combined4)

print("Testing with TypeError (should be caught and move to next validator):")
try:
    TestClass4(x="valid string")
    print("SUCCESS: String value accepted by second validator")
except TypeError as e:
    print(f"TypeError raised: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")