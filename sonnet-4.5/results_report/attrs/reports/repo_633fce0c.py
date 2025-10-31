#!/usr/bin/env python3
"""
Minimal reproduction case for attrs or_ validator exception handling bug.
This demonstrates that or_ catches ALL exceptions instead of just validation exceptions.
"""

import attr
from attr import validators


class BuggyValidator:
    """A validator with a programming error that raises AttributeError."""
    def __call__(self, inst, attr, value):
        # This simulates a programming error in the validator
        raise AttributeError("Bug in validator implementation!")


# Create an or_ validator combining the buggy validator with a normal one
combined = validators.or_(BuggyValidator(), validators.instance_of(str))

@attr.define
class TestClass:
    x: int = attr.field(validator=combined)

# Try to create an instance - the AttributeError should propagate but doesn't
print("Testing or_ validator exception handling with value=42")
print("-" * 60)

try:
    TestClass(x=42)
    print("ERROR: No exception was raised!")
except AttributeError as e:
    print(f"GOOD: AttributeError propagated correctly")
    print(f"  Message: {e}")
except ValueError as e:
    print(f"BUG: or_ hid the AttributeError, raised ValueError instead")
    print(f"  Message: {e}")
except Exception as e:
    print(f"UNEXPECTED: Got {type(e).__name__}: {e}")