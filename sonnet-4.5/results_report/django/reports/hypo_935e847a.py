#!/usr/bin/env python3
"""Hypothesis-based property test for Django check_referrer_policy bug"""

import sys
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.core.checks.security.base import REFERRER_POLICY_VALUES

@given(st.sets(st.sampled_from(list(REFERRER_POLICY_VALUES)), min_size=1, max_size=3))
@settings(max_examples=200)
def test_referrer_policy_trailing_comma(policy_set):
    policy_list = list(policy_set)
    policy_string_with_trailing = ", ".join(policy_list) + ","

    values = {v.strip() for v in policy_string_with_trailing.split(",")}

    assert values <= REFERRER_POLICY_VALUES, \
        f"Trailing comma causes empty string in set: {policy_string_with_trailing!r} -> {values}"

if __name__ == "__main__":
    test_referrer_policy_trailing_comma()