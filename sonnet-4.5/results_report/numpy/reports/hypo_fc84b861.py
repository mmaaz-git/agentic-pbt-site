from hypothesis import given, strategies as st, assume, settings
from pandas.io.excel._util import _range2cols, _excel2num


@given(
    st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=1, max_size=3),
    st.text(alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ', min_size=1, max_size=3)
)
@settings(max_examples=1000)
def test_range2cols_handles_any_range_order(col1, col2):
    """
    Property: _range2cols should handle ranges in any order.
    Either both forward and reversed ranges should work,
    or reversed ranges should raise an error.
    Silently returning an empty list is a bug.
    """
    idx1 = _excel2num(col1)
    idx2 = _excel2num(col2)

    assume(idx1 != idx2)

    forward_range = f"{col1}:{col2}" if idx1 < idx2 else f"{col2}:{col1}"
    reverse_range = f"{col2}:{col1}" if idx1 < idx2 else f"{col1}:{col2}"

    result_forward = _range2cols(forward_range)
    result_reverse = _range2cols(reverse_range)

    min_idx = min(idx1, idx2)
    max_idx = max(idx1, idx2)
    expected_length = max_idx - min_idx + 1

    assert len(result_forward) == expected_length
    assert len(result_reverse) == expected_length


if __name__ == "__main__":
    test_range2cols_handles_any_range_order()