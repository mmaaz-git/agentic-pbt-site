#!/usr/bin/env python3
"""Test parse_cookie with Hypothesis as in the bug report"""

import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        DEFAULT_CHARSET='utf-8',
    )
    django.setup()

from django.http import parse_cookie
from hypothesis import given, strategies as st, settings as hypothesis_settings

@given(st.dictionaries(
    st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'),
        min_codepoint=32, max_codepoint=126
    )),
    st.text(max_size=100, alphabet=st.characters(
        min_codepoint=32, max_codepoint=126
    )),
    min_size=1, max_size=10
))
@hypothesis_settings(max_examples=500)
def test_parse_cookie_preserves_data(cookie_dict):
    cookie_string = "; ".join(f"{k}={v}" for k, v in cookie_dict.items())
    parsed = parse_cookie(cookie_string)

    for key, value in cookie_dict.items():
        assert key in parsed, f"Key {key} not in parsed cookies"
        assert parsed[key] == value, f"Value mismatch for {key}: {parsed[key]} vs {value}"

# Run the test
print("Running Hypothesis test...")
try:
    test_parse_cookie_preserves_data()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
    print("\nThis confirms the bug exists.")