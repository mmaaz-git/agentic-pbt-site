import hypothesis.strategies as st
from hypothesis import given, settings
import pandas.core.computation.parsing as parsing


@given(st.text())
@settings(max_examples=1000)
def test_clean_column_name_idempotent(name):
    try:
        result1 = parsing.clean_column_name(name)
        result2 = parsing.clean_column_name(result1)
        assert result1 == result2
    except Exception as e:
        print(f"Failed on input: {repr(name)}")
        print(f"Exception: {e}")
        raise

if __name__ == "__main__":
    # Run the test
    test_clean_column_name_idempotent()
    print("Test completed")