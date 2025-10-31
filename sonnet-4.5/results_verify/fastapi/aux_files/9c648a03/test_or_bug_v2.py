import attrs
from attrs import validators


def test_or_masks_errors():
    """Test that or_ validator incorrectly masks programming errors."""

    def buggy_validator(inst, attr, value):
        # This should raise NameError
        undefined_variable

    def valid_validator(inst, attr, value):
        pass

    # Create the or_ validator
    v = validators.or_(buggy_validator, valid_validator)

    try:
        # Try to validate - this should raise NameError but doesn't
        v(None, None, 42)
        print("BUG CONFIRMED: No exception raised! The or_ validator masked the NameError.")
        return True
    except NameError as e:
        print(f"NameError properly propagated: {e}")
        return False
    except Exception as e:
        print(f"Other exception raised: {type(e).__name__}: {e}")
        return False


def test_or_with_attrs_define():
    """Test using the reproduction example from the bug report."""

    def buggy_validator(inst, attr, value):
        undefined_variable  # This should raise NameError

    def working_validator(inst, attr, value):
        pass

    try:
        @attrs.define
        class Example:
            value: int = attrs.field(
                validator=validators.or_(buggy_validator, working_validator)
            )

        # Create an instance - should raise NameError but doesn't
        obj = Example(value=42)
        print(f"BUG CONFIRMED: Instance created successfully with value={obj.value}")
        print("The or_ validator masked the NameError in buggy_validator.")
        return True
    except NameError as e:
        print(f"NameError properly propagated: {e}")
        return False
    except Exception as e:
        print(f"Other exception raised: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("Test 1: Direct validator call")
    print("-" * 40)
    bug1 = test_or_masks_errors()

    print("\nTest 2: Using @attrs.define")
    print("-" * 40)
    bug2 = test_or_with_attrs_define()

    print("\n" + "=" * 50)
    if bug1 or bug2:
        print("BUG CONFIRMED: or_ validator masks programming errors")
    else:
        print("Bug not reproduced")