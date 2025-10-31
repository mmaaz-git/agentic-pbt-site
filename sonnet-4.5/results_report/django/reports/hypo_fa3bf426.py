#!/usr/bin/env python3
"""
Hypothesis test that discovers the handle_extensions bug producing invalid '.' extension.
"""
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from django.core.management.utils import handle_extensions

@given(st.lists(st.text(alphabet=' ,', min_size=1, max_size=20), min_size=1, max_size=5))
def test_handle_extensions_no_single_dot(separator_strings):
    result = handle_extensions(separator_strings)
    assert '.' not in result, f"Invalid extension '.' should not be in result, but got {result} from input {separator_strings}"

if __name__ == "__main__":
    # Run the test
    test_handle_extensions_no_single_dot()