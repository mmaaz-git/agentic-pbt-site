"""Test what exceptions are typically raised by attrs validators."""
import attrs
from attrs import validators


# Create a dummy attribute for testing
def make_attr(name="test"):
    @attrs.define
    class Dummy:
        test: str = attrs.field()

    return attrs.fields(Dummy).test


def test_standard_validator_exceptions():
    """Test what exceptions are raised by built-in validators."""

    attr = make_attr()

    # Test instance_of validator
    v_instance = validators.instance_of(int)

    try:
        v_instance(None, attr, "not an int")
    except Exception as e:
        print(f"instance_of raises: {type(e).__name__}: {e}")

    # Test in_ validator
    v_in = validators.in_([1, 2, 3])

    try:
        v_in(None, attr, 4)
    except Exception as e:
        print(f"in_ raises: {type(e).__name__}: {e}")

    # Test gt validator
    v_gt = validators.gt(5)

    try:
        v_gt(None, attr, 3)
    except Exception as e:
        print(f"gt raises: {type(e).__name__}: {e}")

    # Test matches_re validator
    v_re = validators.matches_re(r"^[a-z]+$")

    try:
        v_re(None, attr, "ABC123")
    except Exception as e:
        print(f"matches_re raises: {type(e).__name__}: {e}")

    # Test min_len validator
    v_len = validators.min_len(5)

    try:
        v_len(None, attr, "abc")
    except Exception as e:
        print(f"min_len raises: {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("=== Standard Validator Exceptions ===")
    test_standard_validator_exceptions()