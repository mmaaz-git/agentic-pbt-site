#!/usr/bin/env python3
"""Hypothesis test demonstrating the django.utils.http.is_same_domain bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from django.utils.http import is_same_domain

@given(st.text(min_size=1))
def test_is_same_domain_exact_match(domain):
    """Test that any domain should match itself (identity property)"""
    result = is_same_domain(domain, domain)
    assert result is True, f"Domain '{domain}' doesn't match itself"

if __name__ == "__main__":
    # Run the test
    test_is_same_domain_exact_match()