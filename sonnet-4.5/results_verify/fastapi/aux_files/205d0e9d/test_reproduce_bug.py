#!/usr/bin/env python3
"""Test to reproduce the or_ validator bug"""

from hypothesis import given, strategies as st
import attr

# First, let's test the Hypothesis test case
class BuggyValidator:
    def __call__(self, inst, attr, value):
        undefined_variable  # This will raise NameError

@given(st.integers())
def test_or_validator_should_not_mask_programming_errors(value):
    validator = attr.validators.or_(
        BuggyValidator(),
        attr.validators.instance_of(int)
    )

    @attr.define
    class TestClass:
        x: int = attr.field(validator=validator)

    TestClass(value)

# Run the hypothesis test
print("Running Hypothesis test...")
try:
    test_or_validator_should_not_mask_programming_errors()
    print("Hypothesis test passed (no NameError raised)")
except NameError as e:
    print(f"Hypothesis test raised NameError as expected: {e}")

print("\n" + "="*60 + "\n")

# Now test the specific reproduction case
print("Running specific reproduction case...")

class BuggyValidator2:
    def __call__(self, inst, attr, value):
        undefined_variable  # This will raise NameError

@attr.define
class TestClass:
    value: int = attr.field(
        validator=attr.validators.or_(
            BuggyValidator2(),
            attr.validators.instance_of(int)
        )
    )

try:
    obj = TestClass(42)
    print(f"Created object with value: {obj.value}")
    print("The NameError from BuggyValidator was silently swallowed!")
except NameError as e:
    print(f"NameError was properly raised: {e}")