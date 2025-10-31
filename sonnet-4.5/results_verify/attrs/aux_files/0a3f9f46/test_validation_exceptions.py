"""Test what exceptions are typically raised by attrs validators."""
import attrs
from attrs import validators


def test_standard_validator_exceptions():
    """Test what exceptions are raised by built-in validators."""

    # Test instance_of validator
    v_instance = validators.instance_of(int)

    try:
        v_instance(None, None, "not an int")
    except Exception as e:
        print(f"instance_of raises: {type(e).__name__}: {e}")

    # Test in_ validator
    v_in = validators.in_([1, 2, 3])

    try:
        v_in(None, None, 4)
    except Exception as e:
        print(f"in_ raises: {type(e).__name__}: {e}")

    # Test gt validator
    v_gt = validators.gt(5)

    try:
        v_gt(None, None, 3)
    except Exception as e:
        print(f"gt raises: {type(e).__name__}: {e}")

    # Test matches_re validator
    v_re = validators.matches_re(r"^[a-z]+$")

    try:
        v_re(None, None, "ABC123")
    except Exception as e:
        print(f"matches_re raises: {type(e).__name__}: {e}")


def test_or_with_validation_exceptions():
    """Test that or_ works correctly with normal validation exceptions."""

    def validator_raises_valueerror(inst, attr, value):
        if value != "A":
            raise ValueError("Must be A")

    def validator_raises_typeerror(inst, attr, value):
        if value != "B":
            raise TypeError("Must be B")

    def validator_accepts_c(inst, attr, value):
        if value != "C":
            raise ValueError("Must be C")

    # Test or_ with ValueError validators
    v_or = validators.or_(
        validator_raises_valueerror,
        validator_raises_typeerror,
        validator_accepts_c
    )

    print("\nTesting or_ with normal validation exceptions:")

    # This should pass (third validator accepts)
    try:
        v_or(None, None, "C")
        print("✓ Value 'C' accepted by third validator")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    # This should pass (first validator accepts)
    try:
        v_or(None, None, "A")
        print("✓ Value 'A' accepted by first validator")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    # This should fail (no validator accepts)
    try:
        v_or(None, None, "D")
        print("✗ Value 'D' should have been rejected")
    except ValueError as e:
        print(f"✓ Value 'D' correctly rejected: {e}")


if __name__ == "__main__":
    print("=== Standard Validator Exceptions ===")
    test_standard_validator_exceptions()

    print("\n=== or_ with Validation Exceptions ===")
    test_or_with_validation_exceptions()