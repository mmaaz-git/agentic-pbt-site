#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.util import capitalize_first_letter
from hypothesis import given, strategies as st, example
import traceback

@given(st.text(min_size=1))
@example('ß')
def test_rest_unchanged(s):
    result = capitalize_first_letter(s)
    assert result[1:] == s[1:], f"For input {repr(s)}, result[1:] = {repr(result[1:])} != s[1:] = {repr(s[1:])}"

@given(st.text(min_size=1))
@example('ß')
def test_first_char_uppercase(s):
    result = capitalize_first_letter(s)
    assert result[0] == s[0].upper(), f"For input {repr(s)}, result[0] = {repr(result[0])} != s[0].upper() = {repr(s[0].upper())}"

print("Running test_rest_unchanged...")
try:
    test_rest_unchanged()
    print("✓ test_rest_unchanged passed")
except AssertionError as e:
    print(f"✗ test_rest_unchanged failed: {e}")
except Exception as e:
    print(f"✗ test_rest_unchanged error: {e}")
    traceback.print_exc()

print("\nRunning test_first_char_uppercase...")
try:
    test_first_char_uppercase()
    print("✓ test_first_char_uppercase passed")
except AssertionError as e:
    print(f"✗ test_first_char_uppercase failed: {e}")
except Exception as e:
    print(f"✗ test_first_char_uppercase error: {e}")
    traceback.print_exc()