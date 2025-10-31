from hypothesis import given, strategies as st, settings, assume
import numpy.f2py.symbolic as sym
import sys

# Increase recursion limit to see if we can trigger RecursionError
sys.setrecursionlimit(100)

@given(st.text(min_size=1, max_size=50, alphabet='abcdefghijklmnopqrstuvwxyz +*-()0123456789'))
@settings(max_examples=500)
def test_fromstring_does_not_crash(s):
    try:
        expr = sym.fromstring(s)
    except (ValueError, KeyError, RecursionError, AttributeError):
        assume(False)

# Test the specific case
print("Testing specific case '(':")
try:
    test_fromstring_does_not_crash('(')
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

# Direct test with lower recursion limit
print("\nDirect test with recursion limit 100:")
try:
    expr = sym.fromstring("(")
    print(f"Parsed: {expr}")
except RecursionError as e:
    print(f"RecursionError raised")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")