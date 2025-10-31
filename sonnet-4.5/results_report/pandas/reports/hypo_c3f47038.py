from hypothesis import given, strategies as st
from pandas.core.computation.parsing import clean_column_name


@given(st.text(min_size=1, max_size=50))
def test_clean_column_name_idempotent(name):
    result1 = clean_column_name(name)
    result2 = clean_column_name(result1)
    assert result1 == result2


if __name__ == "__main__":
    test_clean_column_name_idempotent()