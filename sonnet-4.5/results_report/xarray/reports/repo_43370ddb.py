from xarray.core.formatting_html import collapsible_section

# Test with malicious HTML input
user_input = '<script>alert("XSS")</script>'
html = collapsible_section(name=user_input)

print("Generated HTML:")
print(html)
print("\n" + "="*50 + "\n")

# Check if script tag is present (not escaped)
if '<script>' in html:
    print("WARNING: Unescaped <script> tag found in HTML!")
    print(f"'<script>' in html: {('<script>' in html)}")
else:
    print("Script tag properly escaped or not found")

# Check if it was escaped
if '&lt;script&gt;' in html:
    print("Script tag was properly escaped to &lt;script&gt;")
else:
    print("Script tag was NOT escaped to &lt;script&gt;")

# Verify assertions from bug report
assert '<script>' in html, "Script tag should be present (unescaped)"
assert '&lt;script&gt;' not in html, "Script tag should NOT be escaped"

print("\nBUG CONFIRMED: HTML injection vulnerability exists!")
print("The <script> tag passes through unescaped, allowing potential XSS attacks.")