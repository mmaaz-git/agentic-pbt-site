#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings as hyp_settings
from django.utils.http import is_same_domain

print("Testing the hypothesis test case...")
@given(st.text(min_size=1))
@hyp_settings(max_examples=500)
def test_is_same_domain_case_insensitive(host):
    """Property: Domain matching should be case-insensitive"""
    pattern = host.upper()
    result1 = is_same_domain(host.lower(), pattern)
    result2 = is_same_domain(host.upper(), pattern.lower())
    assert result1 == result2, \
        f"Case sensitivity mismatch: is_same_domain({host.lower()!r}, {pattern!r}) = {result1}, " \
        f"but is_same_domain({host.upper()!r}, {pattern.lower()!r}) = {result2}"

# Run the test
try:
    test_is_same_domain_case_insensitive()
    print("Hypothesis test passed - no issues found")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")

print("\nReproducing specific examples from bug report...")

# Test the specific examples
print(f"is_same_domain('A', 'A') = {is_same_domain('A', 'A')}")
print(f"is_same_domain('example.COM', 'EXAMPLE.com') = {is_same_domain('example.COM', 'EXAMPLE.com')}")
print(f"is_same_domain('Example.Com', 'example.com') = {is_same_domain('Example.Com', 'example.com')}")

# More examples to understand the behavior
print("\nAdditional tests:")
print(f"is_same_domain('a', 'A') = {is_same_domain('a', 'A')}")
print(f"is_same_domain('A', 'a') = {is_same_domain('A', 'a')}")
print(f"is_same_domain('example.com', 'EXAMPLE.COM') = {is_same_domain('example.com', 'EXAMPLE.COM')}")
print(f"is_same_domain('EXAMPLE.COM', 'example.com') = {is_same_domain('EXAMPLE.COM', 'example.com')}")

# Test with subdomains
print("\nSubdomain tests:")
print(f"is_same_domain('foo.example.com', '.EXAMPLE.COM') = {is_same_domain('foo.example.com', '.EXAMPLE.COM')}")
print(f"is_same_domain('FOO.EXAMPLE.COM', '.example.com') = {is_same_domain('FOO.EXAMPLE.COM', '.example.com')}")