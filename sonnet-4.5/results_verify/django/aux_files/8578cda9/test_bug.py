#!/usr/bin/env python3
"""Test the purported bug in django.utils.html.CountsDict"""

import sys
import traceback

# First test: Try the simple failing example
print("Test 1: Simple reproduction")
print("-" * 40)
try:
    from django.utils.html import CountsDict
    cd = CountsDict(word="hello", foo="bar")
    print(f"Success: Created CountsDict with kwargs: {cd}")
    print(f"cd['foo'] = {cd.get('foo', 'NOT FOUND')}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n")

# Second test: Try without extra kwargs (the way it's actually used in Django)
print("Test 2: How Django actually uses CountsDict")
print("-" * 40)
try:
    from django.utils.html import CountsDict
    cd = CountsDict(word="hello")
    print(f"Success: Created CountsDict with just word: {cd}")
    print(f"Word stored: {cd.word}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n")

# Third test: Test the property-based test from the bug report
print("Test 3: Property-based test")
print("-" * 40)
try:
    from hypothesis import given, strategies as st
    from django.utils.html import CountsDict

    @given(st.text(min_size=1), st.dictionaries(st.text(min_size=1), st.integers()))
    def test_countsdict_accepts_kwargs_like_dict(word, kwargs_data):
        cd = CountsDict(word=word, **kwargs_data)
        for key, value in kwargs_data.items():
            assert cd[key] == value, f"Expected cd[{key}] == {value}, got {cd.get(key, 'NOT FOUND')}"
        print(f"✓ Test passed with word='{word}' and kwargs={kwargs_data}")

    # Run a few examples
    test_countsdict_accepts_kwargs_like_dict()
    print("Property-based test completed")

except ImportError:
    print("hypothesis not installed, skipping property-based test")
except Exception as e:
    print(f"Error in property-based test: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n")

# Fourth test: Check if a regular dict would accept the same arguments
print("Test 4: Compare with regular dict behavior")
print("-" * 40)
try:
    # Test regular dict with kwargs
    d = dict(foo="bar", baz=42)
    print(f"Regular dict with kwargs: {d}")

    # Test if CountsDict signature suggests it should work like dict
    print("\nCountsDict signature: __init__(self, *args, word, **kwargs)")
    print("This suggests it should accept kwargs like a regular dict")

except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n")

# Fifth test: Verify the actual bug - the typo in super().__init__ call
print("Test 5: Examining the source code")
print("-" * 40)
import inspect
try:
    source = inspect.getsource(CountsDict.__init__)
    print("Source of CountsDict.__init__:")
    print(source)
    if "*kwargs" in source and "super().__init__(*args, *kwargs)" in source:
        print("\n⚠️  BUG CONFIRMED: Using *kwargs instead of **kwargs in super().__init__()")
        print("   Should be: super().__init__(*args, **kwargs)")
    elif "**kwargs" in source and "super().__init__(*args, **kwargs)" in source:
        print("\n✓ Code looks correct: Using **kwargs in super().__init__()")
except Exception as e:
    print(f"Could not inspect source: {e}")