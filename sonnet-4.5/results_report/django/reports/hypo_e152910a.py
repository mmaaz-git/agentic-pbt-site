#!/usr/bin/env python3
"""Hypothesis test for django.template.defaultfilters.get_digit with negative numbers."""

import sys
import os

# Add Django to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
django.setup()

from hypothesis import given, strategies as st
from django.template import defaultfilters


@given(st.integers(min_value=-1000, max_value=-1), st.integers(min_value=1, max_value=10))
def test_get_digit_negative_numbers(value, position):
    result = defaultfilters.get_digit(value, position)
    value_str = str(value)
    try:
        expected = int(value_str[-position])
    except (IndexError, ValueError):
        expected = 0
    assert result == expected or result == value


if __name__ == "__main__":
    # Run the test
    test_get_digit_negative_numbers()