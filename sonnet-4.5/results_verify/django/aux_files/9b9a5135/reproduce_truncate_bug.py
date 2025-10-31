#!/usr/bin/env python3
"""Reproduce the truncate_name bug from the report."""

import os
import sys
import django
from django.conf import settings

# Setup Django
settings.configure(
    DEBUG=True,
    DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}}
)
django.setup()

# Test with hypothesis
print("=== Running Hypothesis Test ===")
from hypothesis import given, strategies as st, settings as hyp_settings
from django.db.backends.utils import truncate_name, split_identifier

@given(st.text(min_size=1), st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=10))
@hyp_settings(max_examples=1000)
def test_truncate_name_length_invariant(identifier, length, hash_len):
    result = truncate_name(identifier, length=length, hash_len=hash_len)

    namespace, name = split_identifier(result)
    name_length = len(name)

    assert name_length <= length, f"Truncated name '{name}' has length {name_length} > {length}"

try:
    test_truncate_name_length_invariant()
    print("Hypothesis test passed (no bugs found)")
except AssertionError as e:
    print(f"Hypothesis test FAILED: {e}")
except Exception as e:
    print(f"Hypothesis test ERROR: {e}")

print("\n=== Reproducing Specific Bug Case ===")
from django.db.backends.utils import truncate_name

identifier = '00'
length = 1
hash_len = 2

result = truncate_name(identifier, length=length, hash_len=hash_len)

print(f"Input: identifier={identifier!r}, length={length}, hash_len={hash_len}")
print(f"Output: {result!r}")
print(f"Output length: {len(result)}")
print(f"Expected: output length <= {length}")
print(f"Actual: output length = {len(result)}")
print(f"Bug confirmed: {len(result) > length}")