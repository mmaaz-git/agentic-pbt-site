from hypothesis import given, strategies as st
import pandas.core.computation.parsing as parsing

@given(st.text())
def test_clean_column_name_always_works(name):
    result = parsing.clean_column_name(name)
    assert result is not None

# Run the test
test_clean_column_name_always_works()