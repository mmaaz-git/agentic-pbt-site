import hypothesis.strategies as st
from hypothesis import given
import pandas.core.computation.parsing as parsing


@given(st.text())
def test_clean_column_name_idempotent(name):
    result1 = parsing.clean_column_name(name)
    result2 = parsing.clean_column_name(result1)
    assert result1 == result2


if __name__ == "__main__":
    test_clean_column_name_idempotent()