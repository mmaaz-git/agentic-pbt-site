import tokenize
from hypothesis import given, strategies as st
from pandas.core.computation.parsing import clean_column_name


@given(st.text())
def test_clean_column_name_no_crash(name):
    try:
        result = clean_column_name(name)
        assert isinstance(result, type(name))
    except SyntaxError:
        pass
    except tokenize.TokenError:
        raise AssertionError(f"TokenError not caught for input: {name!r}")


if __name__ == "__main__":
    test_clean_column_name_no_crash()