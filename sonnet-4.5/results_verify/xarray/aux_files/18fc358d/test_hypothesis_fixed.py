#!/usr/bin/env python3
"""Property-based test for HTML injection in collapsible_section"""

from hypothesis import given, strategies as st, example
from xarray.core.formatting_html import collapsible_section

# Define the property test
@given(st.text())
@example('<script>alert("XSS")</script>')  # Add the specific failing example
def test_collapsible_section_escapes_html_in_name(user_input):
    html = collapsible_section(user_input)
    if '<script>' in user_input:
        assert '<script>' not in html or '&lt;script&gt;' in html, f"Script tag not escaped in: {html[:100]}..."

# Test with the specific failing input
print("Testing with the specific failing input from bug report:")
user_input = '<script>alert("XSS")</script>'
html = collapsible_section(name=user_input)
try:
    if '<script>' in user_input:
        assert '<script>' not in html or '&lt;script&gt;' in html
    print("Test passed")
except AssertionError:
    print(f"Test FAILED with input: '<script>alert(\"XSS\")</script>'")
    print(f"HTML contains unescaped script tag")

# Run property-based testing
print("\nRunning property-based tests...")
try:
    test_collapsible_section_escapes_html_in_name()
    print("All property-based tests passed!")
except AssertionError as e:
    print(f"Property-based test failed: {e}")