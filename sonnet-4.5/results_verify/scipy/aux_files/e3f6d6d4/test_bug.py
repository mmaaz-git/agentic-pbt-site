#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

from html import escape
from hypothesis import given, strategies as st, example
from xarray.core.formatting_html import collapsible_section

# First, test the hypothesis test
@given(st.text(min_size=1, max_size=100))
@example("<script>alert('xss')</script>")
@example("Name&Test")
def test_collapsible_section_escapes_name(name):
    """
    Property: collapsible_section should escape HTML special characters in name.
    Evidence: This function takes arbitrary text and embeds it in HTML.
    """
    result = collapsible_section(name)

    if '<script>' in name:
        assert '<script>' not in result, \
            "Unescaped '<script>' found in output"

    if '&' in name and '&' not in ['&amp;', '&lt;', '&gt;', '&quot;', '&#39;']:
        # Check if standalone & is escaped
        assert '&' not in result or '&amp;' in result, \
            "Unescaped '&' found in output"

# Run the hypothesis test
print("Running hypothesis test...")
try:
    test_collapsible_section_escapes_name()
    print("Hypothesis test passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
except Exception as e:
    print(f"Hypothesis test error: {e}")

# Now run the specific reproduction example
print("\n" + "="*50)
print("Running specific reproduction test...")
print("="*50)

result = collapsible_section("<script>alert('xss')</script>")
print("\nResult of collapsible_section('<script>alert('xss')</script>'):")
print(result)

print("\n" + "="*50)
print("Checking assertions...")
if '<script>' in result:
    print("✗ ASSERTION CONFIRMED: '<script>' IS present in the output (not escaped)")
else:
    print("✓ '<script>' is NOT present in the output (properly escaped)")

# Also test with other HTML characters
print("\n" + "="*50)
print("Testing other HTML characters...")
test_cases = [
    ("Name&Test", "&"),
    ("<div>Test</div>", "<div>"),
    ("Test'>alert('xss')", "'>"),
    ('Test">alert("xss")', '">'),
]

for test_input, check_str in test_cases:
    result = collapsible_section(test_input)
    if check_str in result:
        print(f"✗ '{check_str}' from input '{test_input}' is present in output (not escaped)")
    else:
        print(f"✓ '{check_str}' from input '{test_input}' is NOT in output (escaped or transformed)")