#!/usr/bin/env python3
"""Property-based test for django.db.backends.utils.truncate_name bug"""

import sys
import os
# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.db.backends.utils import truncate_name, split_identifier

@given(st.text(), st.integers(min_value=1, max_value=200))
def test_truncate_name_length_invariant(identifier, length):
    result = truncate_name(identifier, length)
    namespace, name = split_identifier(result)
    if namespace:
        assert len(name) <= length, f"Name portion {name!r} has length {len(name)} > {length}"
    else:
        assert len(result) <= length, f"Result {result!r} has length {len(result)} > {length}"

# Run the test
if __name__ == "__main__":
    # Use explicit settings for reproducibility
    test_truncate_name_length_invariant()