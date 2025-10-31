"""Test the Hypothesis property test from the bug report"""

from hypothesis import given, strategies as st
from pydantic.deprecated.decorator import to_pascal


@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1))
def test_to_pascal_idempotent(snake):
    first = to_pascal(snake)
    second = to_pascal(first)
    assert first == second, f"to_pascal should be idempotent: to_pascal({snake!r}) = {first!r}, to_pascal({first!r}) = {second!r}"

# Run the test
if __name__ == "__main__":
    import sys
    try:
        # Try to find a failing example
        test_to_pascal_idempotent()
        print("No failure found after running hypothesis test")
    except AssertionError as e:
        print(f"Test failed with: {e}")
        sys.exit(1)