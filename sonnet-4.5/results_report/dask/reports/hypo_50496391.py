from hypothesis import given, strategies as st
import math

def _adjust_split_out_for_group_keys(npartitions, by):
    if len(by) == 1:
        return math.ceil(npartitions / 15)
    return math.ceil(npartitions / (10 / (len(by) - 1)))

@given(
    st.integers(min_value=1, max_value=1000),
    st.lists(st.text(), max_size=10)
)
def test_split_out_is_positive(npartitions, by):
    result = _adjust_split_out_for_group_keys(npartitions, by)
    assert result > 0, f"Expected positive split_out, got {result}"

if __name__ == "__main__":
    # Run the test
    test_split_out_is_positive()