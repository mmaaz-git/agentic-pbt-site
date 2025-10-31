#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.utils.translation import to_locale

# Test the basic reproduction case
print("Testing basic reproduction case:")
print(f"to_locale('ENUS') = {repr(to_locale('ENUS'))}")
print(f"to_locale('FRCA') = {repr(to_locale('FRCA'))}")
print(f"to_locale('DEDE') = {repr(to_locale('DEDE'))}")
print(f"to_locale('AAAA') = {repr(to_locale('AAAA'))}")

# Test the property-based test
print("\nTesting the hypothesis test:")
from hypothesis import given, strategies as st

@given(st.text(min_size=4, max_size=10, alphabet=st.characters(min_codepoint=65, max_codepoint=90)))
def test_to_locale_without_dash_should_be_lowercase(language_str):
    result = to_locale(language_str)
    assert result == result.lower(), f"to_locale({language_str!r}) = {result!r}, but should be all lowercase when no dash present"

try:
    test_to_locale_without_dash_should_be_lowercase()
    print("Hypothesis test passed")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")