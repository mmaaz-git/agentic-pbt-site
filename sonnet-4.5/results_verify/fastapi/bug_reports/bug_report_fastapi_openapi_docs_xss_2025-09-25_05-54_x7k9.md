# Bug Report: fastapi.openapi.docs XSS Vulnerabilities

**Target**: `fastapi.openapi.docs.get_swagger_ui_html` and `fastapi.openapi.docs.get_redoc_html`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The HTML generation functions in `fastapi.openapi.docs` do not properly escape user-controlled inputs, allowing Cross-Site Scripting (XSS) attacks through the `title` and `openapi_url` parameters.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
import re

@given(
    openapi_url=st.text(min_size=1, max_size=100),
    title=st.text(min_size=1, max_size=100)
)
@settings(max_examples=500)
def test_swagger_ui_html_xss_injection_attempts(openapi_url, title):
    result = get_swagger_ui_html(openapi_url=openapi_url, title=title)
    html_content = result.body.decode('utf-8')

    dangerous_patterns = [
        r'<script[^>]*>(?!.*SwaggerUIBundle)',
        r'javascript:',
        r'onerror\s*=',
    ]

    for pattern in dangerous_patterns:
        matches = re.findall(pattern, html_content, re.IGNORECASE)
        legitimate_matches = [m for m in matches if 'swagger' in m.lower()]
        suspicious_matches = [m for m in matches if m not in legitimate_matches]

        if suspicious_matches and any(c in openapi_url + title for c in '<>"\''):
            assert False, f"Potential XSS: pattern '{pattern}' found"
```

**Failing input**: `title='<'` (or any input containing HTML/JavaScript special characters)

## Reproducing the Bug

```python
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

title_xss = '<script>alert("XSS")</script>'
result = get_swagger_ui_html(
    openapi_url="https://example.com/openapi.json",
    title=title_xss
)
html = result.body.decode('utf-8')
print(html)

url_xss = "https://example.com'; alert('XSS'); var x='"
result2 = get_swagger_ui_html(
    openapi_url=url_xss,
    title="Safe Title"
)
html2 = result2.body.decode('utf-8')
print(html2)

redoc_title_xss = '<script>alert("XSS in ReDoc")</script>'
result3 = get_redoc_html(
    openapi_url="https://example.com/openapi.json",
    title=redoc_title_xss
)
html3 = result3.body.decode('utf-8')
print(html3)

redoc_url_xss = 'https://example.com"><script>alert("XSS")</script><redoc spec-url="'
result4 = get_redoc_html(
    openapi_url=redoc_url_xss,
    title="Safe Title"
)
html4 = result4.body.decode('utf-8')
print(html4)
```

## Why This Is A Bug

The functions use Python f-strings to directly interpolate user-controlled inputs into HTML and JavaScript without any escaping:

1. In `get_swagger_ui_html` (lines 117-158):
   - `title` is inserted into `<title>{title}</title>` without HTML escaping
   - `openapi_url` is inserted into JavaScript `url: '{openapi_url}'` without JavaScript escaping

2. In `get_redoc_html` (lines 219-253):
   - `title` is inserted into `<title>{title}</title>` without HTML escaping
   - `openapi_url` is inserted into `<redoc spec-url="{openapi_url}">` without HTML escaping

This allows attackers who can control these parameters to inject arbitrary HTML/JavaScript that will execute in users' browsers. While these functions are typically called by FastAPI internally, applications that allow user control over API documentation settings are vulnerable.

## Fix

```diff
--- a/fastapi/openapi/docs.py
+++ b/fastapi/openapi/docs.py
@@ -1,6 +1,7 @@
 import json
+import html
 from typing import Any, Dict, Optional

 from fastapi.encoders import jsonable_encoder
 from starlette.responses import HTMLResponse
 from typing_extensions import Annotated, Doc
@@ -113,13 +114,13 @@ def get_swagger_ui_html(
     if swagger_ui_parameters:
         current_swagger_ui_parameters.update(swagger_ui_parameters)

+    escaped_title = html.escape(title)
+    escaped_openapi_url = openapi_url.replace("'", "\\'")
+
     html = f"""
     <!DOCTYPE html>
     <html>
     <head>
     <link type="text/css" rel="stylesheet" href="{swagger_css_url}">
     <link rel="shortcut icon" href="{swagger_favicon_url}">
-    <title>{title}</title>
+    <title>{escaped_title}</title>
     </head>
     <body>
     <div id="swagger-ui">
     </div>
     <script src="{swagger_js_url}"></script>
     <!-- `SwaggerUIBundle` is now available on the page -->
     <script>
     const ui = SwaggerUIBundle({{
-        url: '{openapi_url}',
+        url: '{escaped_openapi_url}',
     """

@@ -218,10 +221,11 @@ def get_redoc_html(
     [FastAPI docs for Custom Docs UI Static Assets (Self-Hosting)](https://fastapi.tiangolo.com/how-to/custom-docs-ui-assets/).
     """
+    escaped_title = html.escape(title)
+    escaped_openapi_url = html.escape(openapi_url, quote=True)
+
     html = f"""
     <!DOCTYPE html>
     <html>
     <head>
-    <title>{title}</title>
+    <title>{escaped_title}</title>
     <!-- needed for adaptive design -->
     <meta charset="utf-8"/>
     <meta name="viewport" content="width=device-width, initial-scale=1">
@@ -245,7 +249,7 @@ def get_redoc_html(
     <noscript>
         ReDoc requires Javascript to function. Please enable it to browse the documentation.
     </noscript>
-    <redoc spec-url="{openapi_url}"></redoc>
+    <redoc spec-url="{escaped_openapi_url}"></redoc>
     <script src="{redoc_js_url}"> </script>
     </body>
     </html>
```