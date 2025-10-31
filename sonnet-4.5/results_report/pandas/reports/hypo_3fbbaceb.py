from hypothesis import given, strategies as st
import pandas.api.types as pat

@given(st.text())
def test_is_re_compilable(text):
    result = pat.is_re_compilable(text)
    if result:
        import re
        try:
            re.compile(text)
        except Exception as e:
            raise AssertionError(f"is_re_compilable returned True but re.compile failed: {e}")

# Run the test
if __name__ == "__main__":
    test_is_re_compilable()