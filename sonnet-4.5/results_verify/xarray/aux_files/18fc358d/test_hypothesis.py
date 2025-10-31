#!/usr/bin/env python3
"""Property-based test for HTML injection in collapsible_section"""

from hypothesis import given, strategies as st
from xarray.core.formatting_html import collapsible_section

@given(st.text())
def test_collapsible_section_escapes_html_in_name(user_input):
    html = collapsible_section(user_input)
    if '<script>' in user_input:
        assert '<script>' not in html or '&lt;script&gt;' in html

# Run the test with the failing input
print("Testing with the specific failing input from bug report:")
try:
    test_collapsible_section_escapes_html_in_name('<script>alert("XSS")</script>')
    print("Test passed")
except AssertionError as e:
    print(f"Test FAILED with input: '<script>alert(\"XSS\")</script>'")
    print(f"AssertionError: {e}")

# Run property-based testing
print("\nRunning property-based tests...")
try:
    test_collapsible_section_escapes_html_in_name()
    print("All property-based tests passed!")
except Exception as e:
    print(f"Property-based test failed: {e}")