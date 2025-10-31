#!/usr/bin/env python3
"""Test the HTML injection bug in xarray.core.formatting_html.collapsible_section"""

from xarray.core.formatting_html import collapsible_section

# Test with malicious input
user_input = '<script>alert("XSS")</script>'
html = collapsible_section(name=user_input)

print("Generated HTML:")
print(html)
print("\n" + "="*50 + "\n")

# Check if script tag is present unescaped
if '<script>' in html:
    print("❌ BUG CONFIRMED: <script> tag is present unescaped in HTML")
else:
    print("✓ <script> tag is not present unescaped")

if '&lt;script&gt;' in html:
    print("✓ <script> tag is properly escaped as &lt;script&gt;")
else:
    print("❌ <script> tag is NOT escaped")

# Verify the assertions from the bug report
try:
    assert '<script>' in html
    print("\n✓ First assertion passed: '<script>' is in html")
except AssertionError:
    print("\n❌ First assertion failed: '<script>' is NOT in html")

try:
    assert '&lt;script&gt;' not in html
    print("✓ Second assertion passed: '&lt;script&gt;' is NOT in html")
except AssertionError:
    print("❌ Second assertion failed: '&lt;script&gt;' IS in html")