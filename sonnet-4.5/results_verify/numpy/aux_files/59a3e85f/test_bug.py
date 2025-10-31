#!/usr/bin/env python3
"""Test the numpy.f2py.crackfortran.markoutercomma bug"""

from hypothesis import given, strategies as st, settings
import numpy.f2py.crackfortran as crackfortran


@given(st.text())
@settings(max_examples=1000)
def test_markoutercomma_should_not_crash(line):
    try:
        result = crackfortran.markoutercomma(line)
        print(f"Success: markoutercomma({line!r}) = {result!r}")
    except AssertionError as e:
        print(f"AssertionError on input {line!r}: {e}")
        assert False, f"Crashed with AssertionError: {e}"
    except Exception as e:
        print(f"Other exception on input {line!r}: {type(e).__name__}: {e}")

if __name__ == "__main__":
    # Run hypothesis test
    print("Running Hypothesis test...")
    try:
        test_markoutercomma_should_not_crash()
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Test the specific failing case
    print("\n\nTesting specific failing input ')':")
    try:
        result = crackfortran.markoutercomma(')')
        print(f"Result: {result}")
    except AssertionError as e:
        print(f"AssertionError: {e}")
    except Exception as e:
        print(f"Other exception: {type(e).__name__}: {e}")