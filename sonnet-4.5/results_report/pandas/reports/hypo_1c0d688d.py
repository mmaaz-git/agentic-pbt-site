from pandas.core import ops
from hypothesis import given, strategies as st

@given(st.booleans(), st.booleans())
def test_kleene_and_without_mask_equals_regular_and(a, b):
    result = ops.kleene_and(a, b, None, None)
    expected = a and b
    assert result == expected

if __name__ == "__main__":
    test_kleene_and_without_mask_equals_regular_and()