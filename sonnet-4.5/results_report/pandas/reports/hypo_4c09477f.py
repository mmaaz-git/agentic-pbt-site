from hypothesis import given, strategies as st, assume
import string
from pandas.io.excel._util import _excel2num, _range2cols

@given(st.text(alphabet=string.ascii_uppercase + ',:', min_size=1, max_size=20))
def test_range2cols_sorted_and_unique(range_str):
    try:
        result = _range2cols(range_str)
        assert result == sorted(result), f"Result {result} is not sorted"
        assert len(result) == len(set(result)), f"Result {result} contains duplicates"
    except (ValueError, IndexError):
        pass

# Run the test
if __name__ == "__main__":
    test_range2cols_sorted_and_unique()