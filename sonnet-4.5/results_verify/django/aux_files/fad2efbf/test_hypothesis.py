#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from django.utils.http import is_same_domain

@given(st.text(min_size=1))
def test_is_same_domain_exact_match(domain):
    result = is_same_domain(domain, domain)
    assert result is True, f"Failed for domain='{domain}': is_same_domain('{domain}', '{domain}') returned {result}"

# Run the test
print("Running hypothesis test...")
try:
    test_is_same_domain_exact_match()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Test error: {e}")