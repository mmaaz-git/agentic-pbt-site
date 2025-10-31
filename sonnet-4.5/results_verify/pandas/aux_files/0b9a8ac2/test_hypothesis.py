from hypothesis import given, assume, strategies as st
from pandas.io.excel._util import fill_mi_header

@given(
    st.lists(st.integers(min_value=1, max_value=100), min_size=2, max_size=20),
    st.lists(st.booleans(), min_size=2, max_size=20)
)
def test_fill_mi_header_forward_fill_semantics(row, control_row):
    assume(len(row) == len(control_row))

    row_with_blanks = row.copy()
    for i in range(1, len(row_with_blanks), 3):
        row_with_blanks[i] = None

    result_row, _ = fill_mi_header(row_with_blanks, control_row.copy())

    for i, val in enumerate(result_row):
        assert val is not None, f"Found None at position {i}, row={row_with_blanks}, control_row={control_row}"

# Run the test
if __name__ == "__main__":
    test_fill_mi_header_forward_fill_semantics()