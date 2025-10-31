#!/usr/bin/env python
"""
Hypothesis-based property test for Django's validate_file_name function.
This test verifies that the function should reject backslashes consistently.
"""

import os
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, settings, strategies as st
from django.core.files.utils import validate_file_name
from django.core.exceptions import SuspiciousFileOperation

@given(st.text(alphabet=st.characters(blacklist_categories=['Cc', 'Cs']), min_size=1, max_size=100))
@settings(max_examples=1000)
def test_validate_rejects_backslash_as_separator(name):
    if '\\' in name and os.path.basename(name) not in {"", ".", ".."}:
        try:
            validate_file_name(name, allow_relative_path=False)
            assert False, f"Should reject backslash in filename: {name!r}"
        except SuspiciousFileOperation:
            pass

if __name__ == "__main__":
    test_validate_rejects_backslash_as_separator()