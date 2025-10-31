#!/usr/bin/env python3
"""Test script to reproduce the quote_etag idempotence bug"""

import sys
import os

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Set up Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

from hypothesis import given, strategies as st
from django.utils.http import quote_etag

# Property-based test
@given(st.text(min_size=1, max_size=100))
def test_quote_etag_idempotent(etag):
    quoted_once = quote_etag(etag)
    quoted_twice = quote_etag(quoted_once)
    assert quoted_once == quoted_twice, f"Failed for input: {etag!r}, once: {quoted_once!r}, twice: {quoted_twice!r}"

# Manual reproduction
def reproduce_bug():
    print("=== Manual reproduction of the bug ===")
    etag = '"'
    print(f"Input:         {etag!r}")

    result1 = quote_etag(etag)
    print(f"After 1 call:  {result1!r}")

    result2 = quote_etag(result1)
    print(f"After 2 calls: {result2!r}")

    result3 = quote_etag(result2)
    print(f"After 3 calls: {result3!r}")

    print(f"\nIdempotence check: {result1 == result2}")
    return result1 == result2

if __name__ == "__main__":
    # First run the manual reproduction
    is_idempotent = reproduce_bug()

    print("\n=== Running property-based test ===")
    try:
        test_quote_etag_idempotent()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nTrying to find the minimal failing example...")
        # Try the specific case mentioned in the bug report
        try:
            test_quote_etag_idempotent('"')
        except AssertionError as e:
            print(f"Confirmed failure for input '\"': {e}")