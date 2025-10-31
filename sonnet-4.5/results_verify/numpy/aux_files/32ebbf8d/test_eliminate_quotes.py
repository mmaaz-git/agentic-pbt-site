#!/usr/bin/env python3
from hypothesis import given, strategies as st, example
import numpy.f2py.symbolic as symbolic

@given(st.text(min_size=1, max_size=100))
@example("'")  # Known failing case
def test_eliminate_insert_quotes_roundtrip(s):
    try:
        s_no_quotes, d = symbolic.eliminate_quotes(s)
        s_restored = symbolic.insert_quotes(s_no_quotes, d)
        assert s == s_restored
        print(f"✓ Passed for input: {repr(s)}")
    except AssertionError as e:
        print(f"✗ AssertionError for input: {repr(s)}")
        print(f"  Error: {e}")
        raise
    except Exception as e:
        print(f"✗ Other exception for input: {repr(s)}")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error: {e}")
        raise

if __name__ == "__main__":
    # Run the test
    test_eliminate_insert_quotes_roundtrip()