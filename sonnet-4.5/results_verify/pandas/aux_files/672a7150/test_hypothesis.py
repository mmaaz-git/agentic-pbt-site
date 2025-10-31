from hypothesis import given, strategies as st
from pandas.core.computation.parsing import clean_column_name

@given(st.text(min_size=1, max_size=50))
def test_clean_column_name_returns_hashable(name):
    """clean_column_name should return a hashable value."""
    result = clean_column_name(name)
    hash(result)

if __name__ == "__main__":
    test_clean_column_name_returns_hashable()