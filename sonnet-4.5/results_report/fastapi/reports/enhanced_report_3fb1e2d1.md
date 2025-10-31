# Bug Report: fastapi.openapi.docs XSS Vulnerability in HTML Documentation Functions

**Target**: `fastapi.openapi.docs.get_swagger_ui_html` and `fastapi.openapi.docs.get_redoc_html`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_swagger_ui_html` and `get_redoc_html` functions in FastAPI's openapi.docs module do not escape HTML special characters in user-provided parameters, allowing Cross-Site Scripting (XSS) attacks through injection of malicious JavaScript code in the `title` and `openapi_url` parameters.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for FastAPI openapi.docs XSS vulnerability
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

@settings(max_examples=500)
@given(st.text(min_size=1, max_size=100), st.text(min_size=1, max_size=100))
def test_swagger_ui_html_escapes_title_properly(openapi_url, title):
    response = get_swagger_ui_html(openapi_url=openapi_url, title=title)
    content = response.body.decode('utf-8')

    title_tag_start = content.find('<title>')
    title_tag_end = content.find('</title>')

    if title_tag_start != -1 and title_tag_end != -1:
        actual_title = content[title_tag_start + 7:title_tag_end]

        if '<' in title or '>' in title:
            if '<' in actual_title or '>' in actual_title:
                assert False, f"Title not properly escaped. Got {actual_title!r} from input {title!r}"

if __name__ == "__main__":
    test_swagger_ui_html_escapes_title_properly()
```

<details>

<summary>
**Failing input**: `title='<'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 29, in <module>
    test_swagger_ui_html_escapes_title_properly()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 13, in test_swagger_ui_html_escapes_title_properly
    @given(st.text(min_size=1, max_size=100), st.text(min_size=1, max_size=100))
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/13/hypo.py", line 26, in test_swagger_ui_html_escapes_title_properly
    assert False, f"Title not properly escaped. Got {actual_title!r} from input {title!r}"
           ^^^^^
AssertionError: Title not properly escaped. Got '<' from input '<'
Falsifying example: test_swagger_ui_html_escapes_title_properly(
    openapi_url='0',  # or any other generated value
    title='<',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/13/hypo.py:25
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction case demonstrating XSS vulnerability in FastAPI's
get_swagger_ui_html and get_redoc_html functions.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

print("=" * 60)
print("Testing XSS vulnerability in FastAPI openapi.docs functions")
print("=" * 60)

# Test 1: XSS in Swagger UI title
print("\n1. Testing Swagger UI title XSS:")
xss_title = '<script>alert("XSS")</script>'
response = get_swagger_ui_html(openapi_url="/openapi.json", title=xss_title)
content = response.body.decode('utf-8')

# Check if the script tag appears unescaped in the HTML
if xss_title in content:
    print(f"   ✗ VULNERABLE: Raw script tag found in HTML")
    print(f"   Input: {xss_title}")
    # Extract the actual title tag content
    title_start = content.find('<title>')
    title_end = content.find('</title>')
    if title_start != -1 and title_end != -1:
        actual_title = content[title_start + 7:title_end]
        print(f"   Output in <title>: {actual_title}")
else:
    print(f"   ✓ Safe: Script tag was escaped")

# Test 2: XSS in Swagger UI openapi_url
print("\n2. Testing Swagger UI openapi_url XSS:")
xss_url = '"><script>alert("XSS")</script>'
response = get_swagger_ui_html(openapi_url=xss_url, title="Test")
content = response.body.decode('utf-8')

if 'alert("XSS")' in content:
    print(f"   ✗ VULNERABLE: Script injection successful in URL parameter")
    print(f"   Input: {xss_url}")
    # Find the url line in the JavaScript
    url_line_start = content.find("url: '")
    if url_line_start != -1:
        url_line_end = content.find("',", url_line_start)
        if url_line_end != -1:
            actual_url_line = content[url_line_start:url_line_end + 2]
            print(f"   Output in JavaScript: {actual_url_line}")
else:
    print(f"   ✓ Safe: URL was escaped")

# Test 3: XSS in ReDoc title
print("\n3. Testing ReDoc title XSS:")
response = get_redoc_html(openapi_url="/openapi.json", title=xss_title)
content = response.body.decode('utf-8')

if xss_title in content:
    print(f"   ✗ VULNERABLE: Raw script tag found in HTML")
    print(f"   Input: {xss_title}")
    # Extract the actual title tag content
    title_start = content.find('<title>')
    title_end = content.find('</title>')
    if title_start != -1 and title_end != -1:
        actual_title = content[title_start + 7:title_end]
        print(f"   Output in <title>: {actual_title}")
else:
    print(f"   ✓ Safe: Script tag was escaped")

# Test 4: XSS in ReDoc openapi_url
print("\n4. Testing ReDoc openapi_url XSS:")
response = get_redoc_html(openapi_url=xss_url, title="Test")
content = response.body.decode('utf-8')

if 'alert("XSS")' in content:
    print(f"   ✗ VULNERABLE: Script injection successful in URL parameter")
    print(f"   Input: {xss_url}")
    # Find the redoc spec-url attribute
    redoc_start = content.find('<redoc spec-url="')
    if redoc_start != -1:
        redoc_end = content.find('"></redoc>', redoc_start)
        if redoc_end != -1:
            actual_redoc = content[redoc_start:redoc_end + 10]
            print(f"   Output in HTML: {actual_redoc}")
else:
    print(f"   ✓ Safe: URL was escaped")

# Test 5: Simple angle bracket test
print("\n5. Testing simple angle brackets:")
test_title = "Test < and > Characters"
response = get_swagger_ui_html(openapi_url="/openapi.json", title=test_title)
content = response.body.decode('utf-8')

if test_title in content:
    print(f"   ✗ VULNERABLE: Angle brackets not escaped")
    print(f"   Input: {test_title}")
    title_start = content.find('<title>')
    title_end = content.find('</title>')
    if title_start != -1 and title_end != -1:
        actual_title = content[title_start + 7:title_end]
        print(f"   Output in <title>: {actual_title}")
else:
    print(f"   ✓ Safe: Angle brackets were escaped")

print("\n" + "=" * 60)
print("Summary: The functions do not escape HTML special characters,")
print("allowing for Cross-Site Scripting (XSS) attacks.")
print("=" * 60)
```

<details>

<summary>
XSS vulnerability confirmed in all test vectors
</summary>
```
============================================================
Testing XSS vulnerability in FastAPI openapi.docs functions
============================================================

1. Testing Swagger UI title XSS:
   ✗ VULNERABLE: Raw script tag found in HTML
   Input: <script>alert("XSS")</script>
   Output in <title>: <script>alert("XSS")</script>

2. Testing Swagger UI openapi_url XSS:
   ✗ VULNERABLE: Script injection successful in URL parameter
   Input: "><script>alert("XSS")</script>
   Output in JavaScript: url: '"><script>alert("XSS")</script>',

3. Testing ReDoc title XSS:
   ✗ VULNERABLE: Raw script tag found in HTML
   Input: <script>alert("XSS")</script>
   Output in <title>: <script>alert("XSS")</script>

4. Testing ReDoc openapi_url XSS:
   ✗ VULNERABLE: Script injection successful in URL parameter
   Input: "><script>alert("XSS")</script>
   Output in HTML: <redoc spec-url=""><script>alert("XSS")</script>"></redoc>

5. Testing simple angle brackets:
   ✗ VULNERABLE: Angle brackets not escaped
   Input: Test < and > Characters
   Output in <title>: Test < and > Characters

============================================================
Summary: The functions do not escape HTML special characters,
allowing for Cross-Site Scripting (XSS) attacks.
============================================================
```
</details>

## Why This Is A Bug

This is a security vulnerability that violates fundamental web security principles. The functions directly interpolate user-controlled input into HTML and JavaScript contexts using Python f-strings without any HTML escaping, allowing injection of arbitrary HTML and JavaScript code.

**Specific vulnerabilities identified:**

1. **In `get_swagger_ui_html` (line 123)**: The `title` parameter is inserted directly into `<title>{title}</title>` without escaping, allowing HTML injection
2. **In `get_swagger_ui_html` (line 132)**: The `openapi_url` parameter is inserted into JavaScript context `url: '{openapi_url}'` without escaping, allowing script injection
3. **In `get_redoc_html` (line 223)**: The `title` parameter is inserted directly into `<title>{title}</title>` without escaping
4. **In `get_redoc_html` (line 248)**: The `openapi_url` parameter is inserted into HTML attribute `<redoc spec-url="{openapi_url}"></redoc>` without escaping

**Security Best Practices Violated:**
- OWASP guidelines require all user-controlled input to be escaped when rendered in HTML
- Modern web frameworks should be secure-by-default
- HTML-generating functions must escape special characters to prevent XSS
- The functions' documentation does not warn about XSS risks or state that input should be pre-escaped

## Relevant Context

FastAPI is a modern, high-performance web framework that emphasizes security and best practices. These functions are part of the public API used to generate interactive API documentation pages (Swagger UI at `/docs` and ReDoc at `/redoc`).

While typical FastAPI applications use hardcoded values for these parameters (coming from application configuration), applications that allow customization through:
- Environment variables
- Configuration files
- Admin panels or settings pages
- Multi-tenant scenarios with user-specific documentation

...would be vulnerable to stored XSS attacks if user-controlled values are passed to these functions.

**Relevant Documentation:**
- FastAPI docs for these functions: https://fastapi.tiangolo.com/reference/openapi/docs/
- The functions are located in: `/fastapi/openapi/docs.py`
- OWASP XSS Prevention Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html

## Proposed Fix

```diff
diff --git a/fastapi/openapi/docs.py b/fastapi/openapi/docs.py
index 1234567..abcdefg 100644
--- a/fastapi/openapi/docs.py
+++ b/fastapi/openapi/docs.py
@@ -1,4 +1,5 @@
 import json
+import html
 from typing import Any, Dict, Optional

 from fastapi.encoders import jsonable_encoder
@@ -120,7 +121,7 @@ def get_swagger_ui_html(
     <head>
     <link type="text/css" rel="stylesheet" href="{swagger_css_url}">
     <link rel="shortcut icon" href="{swagger_favicon_url}">
-    <title>{title}</title>
+    <title>{html.escape(title)}</title>
     </head>
     <body>
     <div id="swagger-ui">
@@ -129,7 +130,7 @@ def get_swagger_ui_html(
     <!-- `SwaggerUIBundle` is now available on the page -->
     <script>
     const ui = SwaggerUIBundle({{
-        url: '{openapi_url}',
+        url: '{html.escape(openapi_url, quote=True)}',
     """

     for key, value in current_swagger_ui_parameters.items():
@@ -220,7 +221,7 @@ def get_redoc_html(
     <!DOCTYPE html>
     <html>
     <head>
-    <title>{title}</title>
+    <title>{html.escape(title)}</title>
     <!-- needed for adaptive design -->
     <meta charset="utf-8"/>
     <meta name="viewport" content="width=device-width, initial-scale=1">
@@ -245,7 +246,7 @@ def get_redoc_html(
     <noscript>
         ReDoc requires Javascript to function. Please enable it to browse the documentation.
     </noscript>
-    <redoc spec-url="{openapi_url}"></redoc>
+    <redoc spec-url="{html.escape(openapi_url, quote=True)}"></redoc>
     <script src="{redoc_js_url}"> </script>
     </body>
     </html>
```