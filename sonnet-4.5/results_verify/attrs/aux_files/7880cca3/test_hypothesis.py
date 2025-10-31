from hypothesis import given, strategies as st
import attr

@given(st.integers())
def test_gt_validator_uses_correct_operator(bound):
    validator = attr.validators.gt(bound)

    @attr.define
    class TestClass:
        value: int = attr.field(validator=validator)

    try:
        TestClass(bound)
        passed_at_bound = True
    except ValueError:
        passed_at_bound = False

    try:
        TestClass(bound + 1)
        passed_above_bound = True
    except ValueError:
        passed_above_bound = False

    assert not passed_at_bound, f"gt({bound}) should reject value == {bound}"
    assert passed_above_bound, f"gt({bound}) should accept value == {bound + 1}"

if __name__ == "__main__":
    test_gt_validator_uses_correct_operator()