#!/usr/bin/env python3

from hypothesis import given, strategies as st
import attr

# First, run the property-based test
@given(st.integers(min_value=-100, max_value=100))
def test_gt_validator_uses_correct_operator(value):
    bound = 50

    @attr.define
    class TestClass:
        x: int = attr.field(validator=attr.validators.gt(bound))

    if value > bound:
        obj = TestClass(x=value)
        assert obj.x == value
    else:
        try:
            TestClass(x=value)
            assert False, f"Should have rejected {value} <= {bound}"
        except ValueError:
            pass

print("Running property-based test...")
test_gt_validator_uses_correct_operator()
print("Property-based test passed!")

# Now run the specific reproduction example
print("\nRunning specific reproduction example...")

@attr.define
class TestClass:
    value: int = attr.field(validator=attr.validators.gt(5))

TestClass(value=6)
print("Created instance with value=6 (should succeed)")

try:
    TestClass(value=5)
    print("BUG: value=5 was accepted (would happen if using operator.ge)")
except ValueError:
    print("Correct: value=5 was rejected (confirms use of operator.gt)")

try:
    TestClass(value=4)
    print("BUG: value=4 was accepted")
except ValueError:
    print("Correct: value=4 was rejected (confirms use of operator.gt)")

print("\nBehavior test complete!")