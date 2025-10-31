from hypothesis import given, strategies as st
import attr
from attr import validators
import operator

@given(st.integers())
def test_gt_uses_correct_operator(value):
    bound = 10
    v = validators.gt(bound)

    if value > bound:
        attr_obj = attr.Attribute(
            name="test", default=None, validator=None, repr=True,
            cmp=None, eq=True, eq_key=None, order=False,
            order_key=None, hash=None, init=True, kw_only=False,
            type=None, converter=None, metadata={}, alias=None
        )
        v(None, attr_obj, value)

    assert "operator.gt" in str(operator.gt)
    assert "operator.ge" in validators.gt.__doc__

if __name__ == "__main__":
    test_gt_uses_correct_operator()
    print("Test completed")