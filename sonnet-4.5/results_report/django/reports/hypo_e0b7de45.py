#!/usr/bin/env python3
"""
Hypothesis property-based test for django.db.backends.sqlite3._functions._sqlite_lpad
Tests the fundamental invariant: LPAD should always return a string of exactly 'length' characters
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.db.backends.sqlite3._functions import _sqlite_lpad

@given(st.text(), st.integers(min_value=0, max_value=10000), st.text(min_size=0))
@settings(max_examples=10000)
def test_lpad_length_invariant(text, length, fill_text):
    result = _sqlite_lpad(text, length, fill_text)
    if result is not None:
        assert len(result) == length, f"Expected length {length}, got {len(result)} for _sqlite_lpad({text!r}, {length}, {fill_text!r})"

if __name__ == "__main__":
    test_lpad_length_invariant()