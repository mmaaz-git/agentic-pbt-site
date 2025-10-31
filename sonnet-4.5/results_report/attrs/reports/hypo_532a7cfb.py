from hypothesis import given, strategies as st
import attr

class BuggyValidator:
    def __call__(self, inst, attr, value):
        undefined_variable

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

# Run the test
test_or_validator_should_not_mask_programming_errors()