from hypothesis import given, strategies as st
from pandas.core.computation.parsing import clean_column_name

@given(st.text(min_size=1, max_size=20))
def test_clean_column_name_returns_hashable(name):
    result = clean_column_name(name)
    hash(result)

# Run the test
if __name__ == "__main__":
    test_clean_column_name_returns_hashable()