import attrs
from attrs import validators
from hypothesis import given, strategies as st


def make_attr():
    return attrs.Attribute(
        name="test", default=None, validator=None, repr=True,
        cmp=None, eq=True, eq_key=None, order=False,
        order_key=None, hash=None, init=True, kw_only=False,
        type=None, converter=None, metadata={}, alias=None
    )


@given(st.integers())
def test_or_should_not_mask_programming_errors(x):
    def buggy_validator(inst, attr, value):
        undefined_variable  # This should raise NameError

    def valid_validator(inst, attr, value):
        pass

    v = validators.or_(buggy_validator, valid_validator)

    # This should raise NameError but doesn't due to the bug
    v(None, make_attr(), x)
    print(f"Test passed with value {x} - no exception raised!")


# Run the test
if __name__ == "__main__":
    test_or_should_not_mask_programming_errors()