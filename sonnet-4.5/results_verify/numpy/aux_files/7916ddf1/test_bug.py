#!/usr/bin/env python3
"""Test to reproduce the eliminate_quotes bug"""

import numpy.f2py.symbolic as symbolic
import traceback

print("Testing numpy.f2py.symbolic.eliminate_quotes with unmatched quotes")
print("=" * 60)

# Test 1: Single double quote
print("\nTest 1: eliminate_quotes('\"')")
try:
    result = symbolic.eliminate_quotes('"')
    print(f"Result: {result}")
except AssertionError as e:
    print("AssertionError raised!")
    traceback.print_exc()
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

# Test 2: Single single quote
print("\nTest 2: eliminate_quotes(\"'\")")
try:
    result = symbolic.eliminate_quotes("'")
    print(f"Result: {result}")
except AssertionError as e:
    print("AssertionError raised!")
    traceback.print_exc()
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

# Test 3: Valid quoted strings
print("\nTest 3: eliminate_quotes('\"hello\"') - valid double quoted")
try:
    result = symbolic.eliminate_quotes('"hello"')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

print("\nTest 4: eliminate_quotes(\"'hello'\") - valid single quoted")
try:
    result = symbolic.eliminate_quotes("'hello'")
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

# Test the property-based test
print("\n" + "=" * 60)
print("Testing the property-based test from the bug report")
from hypothesis import given, strategies as st

@given(st.text())
def test_eliminate_insert_quotes_roundtrip(s):
    new_s, mapping = symbolic.eliminate_quotes(s)
    restored = symbolic.insert_quotes(new_s, mapping)
    assert restored == s

# Run with specific failing examples
print("\nTesting roundtrip with '\"'")
try:
    test_eliminate_insert_quotes_roundtrip.hypothesis.explicit_examples = [('"',)]
    test_eliminate_insert_quotes_roundtrip()
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\nTesting roundtrip with \"'\"")
try:
    test_eliminate_insert_quotes_roundtrip.hypothesis.explicit_examples = [("'",)]
    test_eliminate_insert_quotes_roundtrip()
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")
    traceback.print_exc()