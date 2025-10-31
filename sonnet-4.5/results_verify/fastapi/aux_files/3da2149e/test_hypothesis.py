import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

import attr
from hypothesis import given, strategies as st


class ValidatorWithNameError:
    """A validator with a programming error"""
    def __call__(self, inst, attr_obj, value):
        undefined_variable  # This should raise NameError


@given(st.integers())
def test_or_validator_should_not_mask_name_errors(value):
    """
    Property: or_ validator should not silently catch programming errors.

    Evidence:
    - and_ validator does not catch exceptions (just calls each validator)
    - not_ validator only catches (ValueError, TypeError) by default
    - or_ validator catches ALL exceptions (line 675 in validators.py)
    """
    validator = attr.validators.or_(
        ValidatorWithNameError(),
        attr.validators.instance_of(int)
    )

    field_attr = attr.Attribute(
        name='value',
        default=attr.NOTHING,
        validator=None,
        repr=True,
        cmp=None,
        eq=True,
        eq_key=None,
        order=False,
        order_key=None,
        hash=None,
        init=True,
        metadata={},
        type=None,
        converter=None,
        kw_only=False,
        inherited=False,
        on_setattr=None,
        alias=None
    )

    try:
        validator(None, field_attr, value)
        assert False, "NameError was silently caught"
    except NameError:
        pass


if __name__ == "__main__":
    # Test with a specific value
    try:
        test_or_validator_should_not_mask_name_errors()
        print("Hypothesis test passed")
    except AssertionError as e:
        print(f"Hypothesis test failed: {e}")

    # Also test directly with specific value
    print("\nDirect test with value 42:")
    validator = attr.validators.or_(
        ValidatorWithNameError(),
        attr.validators.instance_of(int)
    )

    field_attr = attr.Attribute(
        name='value',
        default=attr.NOTHING,
        validator=None,
        repr=True,
        cmp=None,
        eq=True,
        eq_key=None,
        order=False,
        order_key=None,
        hash=None,
        init=True,
        metadata={},
        type=None,
        converter=None,
        kw_only=False,
        inherited=False,
        on_setattr=None,
        alias=None
    )

    try:
        validator(None, field_attr, 42)
        print("ERROR: NameError was silently caught - no exception raised!")
    except NameError:
        print("GOOD: NameError was raised as expected")