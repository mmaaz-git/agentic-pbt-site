from hypothesis import given, settings, strategies as st
from pandas.core.computation.parsing import clean_column_name

@given(st.text(min_size=1))
@settings(max_examples=1000)
def test_clean_column_name_idempotent(name):
    try:
        first = clean_column_name(name)
        second = clean_column_name(first)
        assert first == second
    except (SyntaxError, TypeError):
        pass

if __name__ == "__main__":
    test_clean_column_name_idempotent()