import re
from hypothesis import given, strategies as st
import pandas.api.types as pat


@given(st.text())
def test_is_re_compilable_for_strings(x):
    result = pat.is_re_compilable(x)
    if result:
        try:
            re.compile(x)
        except re.error:
            raise AssertionError(f"is_re_compilable({x!r}) returned True but re.compile raised error")


if __name__ == "__main__":
    test_is_re_compilable_for_strings()