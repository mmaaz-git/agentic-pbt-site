from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.io.clipboards import read_clipboard, to_clipboard


@given(st.sampled_from(['latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'ascii']))
@settings(max_examples=10)
def test_both_functions_reject_non_utf8(encoding):
    if encoding.lower().replace('-', '') == 'utf8':
        return

    read_exc_type = None
    write_exc_type = None

    try:
        read_clipboard(encoding=encoding)
    except Exception as e:
        read_exc_type = type(e).__name__

    try:
        to_clipboard(pd.DataFrame([[1, 2]]), encoding=encoding)
    except Exception as e:
        write_exc_type = type(e).__name__

    assert read_exc_type == write_exc_type, (
        f"Inconsistent exception types for encoding '{encoding}': "
        f"read_clipboard raises {read_exc_type}, to_clipboard raises {write_exc_type}"
    )

if __name__ == "__main__":
    test_both_functions_reject_non_utf8()