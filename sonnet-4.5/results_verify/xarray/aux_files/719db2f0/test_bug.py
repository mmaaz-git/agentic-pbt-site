#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from html import escape
from hypothesis import given, strategies as st, example

# First, let's run the hypothesis test
from xarray.core.formatting_html import _icon

print("Testing with Hypothesis:")
@given(st.text(min_size=1, max_size=100))
@example("<script>alert('xss')</script>")
@example("test&name")
def test_icon_escapes_html_characters(icon_name):
    """
    Property: _icon should escape HTML special characters in icon_name.
    Evidence: Other functions in the same file use escape() for user inputs.
    """
    result = _icon(icon_name)

    dangerous_chars = ['<', '>', '&', '"', "'"]
    if any(char in icon_name for char in dangerous_chars):
        escaped_icon_name = escape(str(icon_name))
        assert escaped_icon_name in result or icon_name not in result, \
            f"Icon name should be HTML-escaped. Got: {result}"

# Run the test
try:
    test_icon_escapes_html_characters()
    print("Hypothesis test passed")
except AssertionError as e:
    print(f"Hypothesis test FAILED: {e}")

print("\n" + "="*60 + "\n")

# Now run the specific reproduction case
print("Testing specific case: <script>alert('xss')</script>")
result = _icon("<script>alert('xss')</script>")
print(f"Result: {result}")
print(f"Contains unescaped <script>: {'<script>' in result}")
print(f"Contains unescaped </script>: {'</script>' in result}")

print("\n" + "="*60 + "\n")

# Test another case with ampersand
print("Testing case with ampersand: test&name")
result2 = _icon("test&name")
print(f"Result: {result2}")
print(f"Contains unescaped &: {'test&name' in result2}")
print(f"Contains escaped &amp;: {'test&amp;name' in result2}")

print("\n" + "="*60 + "\n")

# Compare with how other functions handle HTML escaping
print("Comparing with other functions in the same module:")
from xarray.core.formatting_html import format_dims, summarize_attrs

# Test format_dims with HTML characters
test_dims = {"<script>test": 10, "normal": 20}
dims_result = format_dims(test_dims, set())
print(f"format_dims with '<script>test': {dims_result}")
print(f"format_dims escapes HTML: {'&lt;script&gt;' in dims_result}")

# Test summarize_attrs with HTML characters
test_attrs = {"<key>": "<value>"}
attrs_result = summarize_attrs(test_attrs)
print(f"summarize_attrs with HTML chars: {attrs_result}")
print(f"summarize_attrs escapes HTML: {'&lt;' in attrs_result}")