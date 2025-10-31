#!/usr/bin/env python3
"""Hypothesis test for CaseInsensitiveMapping."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, example
from django.utils.datastructures import CaseInsensitiveMapping


@given(st.dictionaries(st.text(), st.text()))
@settings(max_examples=500)
@example({'ß': ''})  # Add the German eszett as a specific example to test
def test_case_insensitive_mapping_access(d):
    cim = CaseInsensitiveMapping(d)
    for key, value in d.items():
        assert cim.get(key) == value
        assert cim.get(key.upper()) == value
        assert cim.get(key.lower()) == value


if __name__ == "__main__":
    # Run the test
    test_case_insensitive_mapping_access()