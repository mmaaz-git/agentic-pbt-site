#!/usr/bin/env python3
"""Test the Unicode variable name bug in scipy.io.matlab.savemat"""

import io
import numpy as np
from scipy.io.matlab import savemat, loadmat

# Test 1: Basic reproduction of the bug
print("Test 1: Basic reproduction with Unicode character 'Ā' (U+0100)")
try:
    f = io.BytesIO()
    data = {'Ā': np.array([[1, 2, 3]])}
    savemat(f, data)
    print("  SUCCESS: No error occurred")
except UnicodeEncodeError as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Test 2: ASCII character (should work)
print("\nTest 2: ASCII character 'A' (U+0041)")
try:
    f = io.BytesIO()
    data = {'A': np.array([[1, 2, 3]])}
    savemat(f, data)
    print("  SUCCESS: No error occurred")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Test 3: Latin1 character within range (should work)
print("\nTest 3: Latin1 character 'ÿ' (U+00FF - last latin1 char)")
try:
    f = io.BytesIO()
    data = {'ÿ': np.array([[1, 2, 3]])}
    savemat(f, data)
    print("  SUCCESS: No error occurred")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

# Test 4: Various Unicode characters outside latin1
unicode_tests = [
    ('Ā', 'U+0100', 'Latin Extended-A'),
    ('中', 'U+4E2D', 'Chinese'),
    ('α', 'U+03B1', 'Greek'),
    ('π', 'U+03C0', 'Greek'),
    ('Д', 'U+0414', 'Cyrillic'),
    ('א', 'U+05D0', 'Hebrew'),
    ('あ', 'U+3042', 'Japanese Hiragana'),
]

print("\nTest 4: Various Unicode characters outside latin1 range")
for char, code, desc in unicode_tests:
    try:
        f = io.BytesIO()
        data = {char: np.array([[42]])}
        savemat(f, data)
        print(f"  {char} ({code} - {desc}): SUCCESS")
    except UnicodeEncodeError as e:
        print(f"  {char} ({code} - {desc}): UnicodeEncodeError")
    except Exception as e:
        print(f"  {char} ({code} - {desc}): {type(e).__name__}")

# Test 5: Run the Hypothesis test from the bug report
print("\nTest 5: Running the property-based test from the bug report")
from hypothesis import given, strategies as st, settings

valid_var_name = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N'), whitelist_characters='_'),
    min_size=1, max_size=31
).filter(lambda x: x[0].isalpha() or x[0] == '_').filter(lambda x: not x.startswith('_'))

@given(
    var_name=valid_var_name,
    data=st.integers(min_value=-1e10, max_value=1e10).map(lambda x: np.array([[x]])),
)
@settings(max_examples=50, deadline=None)
def test_round_trip_savemat_loadmat(var_name, data):
    f = io.BytesIO()
    original_dict = {var_name: data}

    savemat(f, original_dict)
    f.seek(0)

    loaded_dict = loadmat(f)
    assert var_name in loaded_dict

try:
    test_round_trip_savemat_loadmat()
    print("  Property-based test passed all examples")
except Exception as e:
    print(f"  Property-based test failed: {type(e).__name__}")
    if hasattr(e, '__context__') and e.__context__:
        print(f"    Root cause: {type(e.__context__).__name__}: {e.__context__}")