from hypothesis import given, strategies as st
from Cython.Compiler import Options

@given(st.text(min_size=1))
def test_parse_variable_value_no_crash(s):
    try:
        result = Options.parse_variable_value(s)
        print(f"Input: {repr(s)} -> Result: {repr(result)}")
    except Exception as e:
        print(f"Input: {repr(s)} -> Exception: {type(e).__name__}: {e}")
        raise

# Run the test
if __name__ == "__main__":
    test_parse_variable_value_no_crash()