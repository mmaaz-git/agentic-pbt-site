# Bug Report: xarray.core.formatting_html.collapsible_section HTML Injection Vulnerability

**Target**: `xarray.core.formatting_html.collapsible_section`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `collapsible_section` function in xarray's HTML formatting module fails to escape HTML special characters in the `name` parameter, allowing HTML injection that could lead to XSS vulnerabilities when user-controlled input is passed to this function.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.core.formatting_html import collapsible_section

@given(st.text())
def test_collapsible_section_escapes_html_in_name(user_input):
    html = collapsible_section(user_input)
    if '<script>' in user_input:
        assert '<script>' not in html or '&lt;script&gt;' in html

# Run the test
if __name__ == "__main__":
    # This will find the failing case
    try:
        test_collapsible_section_escapes_html_in_name()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed!")
        import traceback
        traceback.print_exc()
```

<details>

<summary>
**Failing input**: `'<script>'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 14, in <module>
    test_collapsible_section_escapes_html_in_name()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 5, in test_collapsible_section_escapes_html_in_name
    def test_collapsible_section_escapes_html_in_name(user_input):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 8, in test_collapsible_section_escapes_html_in_name
    assert '<script>' not in html or '&lt;script&gt;' in html
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_collapsible_section_escapes_html_in_name(
    user_input='<script>',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/30/hypo.py:8
Test failed!
```
</details>

## Reproducing the Bug

```python
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
```

<details>

<summary>
HTML injection vulnerability confirmed - unescaped script tag in output
</summary>
```
Generated HTML:
<input id='section-1aaac813-87aa-4aa4-8271-53578c6f9360' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-1aaac813-87aa-4aa4-8271-53578c6f9360' class='xr-section-summary'  title='Expand/collapse section'><script>alert("XSS")</script>:</label><div class='xr-section-inline-details'></div><div class='xr-section-details'></div>

==================================================

WARNING: Unescaped <script> tag found in HTML!
'<script>' in html: True
Script tag was NOT escaped to &lt;script&gt;

BUG CONFIRMED: HTML injection vulnerability exists!
The <script> tag passes through unescaped, allowing potential XSS attacks.
```
</details>

## Why This Is A Bug

This is a legitimate security vulnerability that violates multiple expectations and established patterns:

1. **Security Vulnerability**: The function allows arbitrary HTML/JavaScript injection through the `name` parameter. When the generated HTML is rendered in a browser, any JavaScript in the name will execute, potentially leading to XSS attacks.

2. **Inconsistent with Module Pattern**: The `formatting_html` module imports `escape` from the `html` module (line 7) and consistently uses it throughout - I found 14 instances of `escape()` being used to sanitize user input before including it in HTML. The `collapsible_section` function is the ONLY function in the entire module that generates HTML without escaping string parameters.

3. **Violates Implicit API Contract**: Functions that generate HTML have an implicit contract to produce safe output. Users reasonably expect that passing arbitrary strings to HTML-generating functions won't create security vulnerabilities.

4. **Public API Exposure**: The function is publicly accessible as `xarray.core.formatting_html.collapsible_section` and can be called by external code, plugins, or user applications that might pass user-controlled input.

5. **No Documentation Warning**: The function has no docstring or documentation warning users that the `name` parameter is not escaped and could be dangerous with untrusted input.

## Relevant Context

The `collapsible_section` function is defined at line 173-192 of `/xarray/core/formatting_html.py`. Analysis of the file shows:

- The module imports `escape` from the standard `html` module at line 7
- Every other function that handles string-to-HTML conversion uses `escape()`:
  - Line 43: `escape(short_data_repr(array))`
  - Line 57: `escape(str(dim))` for dimension names
  - Line 66: `escape(str(k))` and `escape(str(v))` for attributes
  - Line 85: `escape(str(name))` for variable names
  - Line 145: `escape(str(n))` for coordinate names
  - And 9 more instances throughout the file

The function is currently called internally with hardcoded strings like "Dimensions", "Coordinates", "Data variables", "Indexes", and "Attributes" (lines 230, 259, 267, 274, 283), which explains why this vulnerability hasn't caused issues in xarray's own code. However, as a public function, it should handle arbitrary input safely.

GitHub source: https://github.com/pydata/xarray/blob/main/xarray/core/formatting_html.py#L173-L192

## Proposed Fix

The fix is straightforward - add HTML escaping to the `name` parameter to match the pattern used throughout the rest of the module:

```diff
--- a/xarray/core/formatting_html.py
+++ b/xarray/core/formatting_html.py
@@ -186,7 +186,7 @@ def collapsible_section(
         f"<input id='{data_id}' class='xr-section-summary-in' "
         f"type='checkbox' {enabled} {collapsed}>"
         f"<label for='{data_id}' class='xr-section-summary' {tip}>"
-        f"{name}:{n_items_span}</label>"
+        f"{escape(name)}:{n_items_span}</label>"
         f"<div class='xr-section-inline-details'>{inline_details}</div>"
         f"<div class='xr-section-details'>{details}</div>"
     )
```