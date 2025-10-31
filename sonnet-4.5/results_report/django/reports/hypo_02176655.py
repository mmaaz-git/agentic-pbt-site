#!/usr/bin/env python3
"""Hypothesis property-based test for django.utils.http.is_same_domain case sensitivity"""

# Add Django environment to path
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings as hyp_settings
from django.utils.http import is_same_domain

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

if __name__ == "__main__":
    test_is_same_domain_case_insensitive()