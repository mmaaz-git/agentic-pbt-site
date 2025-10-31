# Bug Report: fastapi.openapi HTML/JavaScript Injection Vulnerabilities

**Target**: `fastapi.openapi.docs`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_swagger_ui_html()` and `get_redoc_html()` functions in `fastapi.openapi.docs` are vulnerable to HTML and JavaScript injection attacks due to unescaped user input being directly interpolated into HTML and JavaScript contexts using f-strings.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

@given(title=st.text())
def test_title_should_not_allow_html_injection(title):
    result = get_swagger_ui_html(
        openapi_url="http://example.com/openapi.json",
        title=title,
    )
    html = result.body.decode('utf-8')

    if '<script>' in title.lower() or '</title>' in title.lower():
        assert '<script>' not in html or title not in html.split('<title>')[1].split('</title>')[0]

@given(openapi_url=st.text())
def test_openapi_url_should_not_allow_js_injection(openapi_url):
    result = get_swagger_ui_html(
        openapi_url=openapi_url,
        title="Test",
    )
    html = result.body.decode('utf-8')

    if "'; alert(" in openapi_url:
        assert "'; alert(" not in html or openapi_url not in html
```

**Failing input 1**: `title = "</title><script>alert('XSS')</script><title>"`
**Failing input 2**: `openapi_url = "'; alert('XSS'); //'"`
**Failing input 3**: `title = "</title><script>alert('XSS')</script><title>"` in `get_redoc_html()`

## Reproducing the Bug

```python
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

malicious_title = "</title><script>alert('XSS')</script><title>Fake"
result = get_swagger_ui_html(
    openapi_url="http://example.com/openapi.json",
    title=malicious_title,
)
html = result.body.decode('utf-8')
assert "<script>alert('XSS')</script>" in html

malicious_url = "'; alert('XSS'); //'"
result = get_swagger_ui_html(
    openapi_url=malicious_url,
    title="API",
)
html = result.body.decode('utf-8')
assert "alert('XSS')" in html

result = get_redoc_html(
    openapi_url="http://example.com/openapi.json",
    title=malicious_title,
)
html = result.body.decode('utf-8')
assert "<script>alert('XSS')</script>" in html
```

## Why This Is A Bug

FastAPI's OpenAPI documentation functions accept user-controlled input (title, openapi_url) and directly interpolate them into HTML and JavaScript without proper escaping. This violates fundamental web security principles:

1. **HTML Context Injection** (lines 123, 223 in docs.py): The `title` parameter is inserted directly into `<title>{title}</title>` without HTML escaping. An attacker can close the title tag early and inject arbitrary HTML/JavaScript.

2. **JavaScript Context Injection** (line 133 in docs.py): The `openapi_url` is inserted into JavaScript as `url: '{openapi_url}'` without JavaScript string escaping. An attacker can break out of the string literal and execute arbitrary JavaScript.

3. **Similar issues exist in other URL parameters** (swagger_css_url, swagger_js_url, swagger_favicon_url, etc.): While less likely to contain user input in typical deployments, they use the same unsafe interpolation pattern.

This could allow:
- Cross-Site Scripting (XSS) attacks if any of these parameters come from user input
- Session hijacking, credential theft, and other security compromises
- Defacement of documentation pages

## Fix

```diff
--- a/fastapi/openapi/docs.py
+++ b/fastapi/openapi/docs.py
@@ -1,5 +1,6 @@
 import json
 from typing import Any, Dict, Optional
+import html as html_module

 from fastapi.encoders import jsonable_encoder
 from starlette.responses import HTMLResponse
@@ -117,10 +118,10 @@ def get_swagger_ui_html(
     html = f"""
     <!DOCTYPE html>
     <html>
     <head>
-    <link type="text/css" rel="stylesheet" href="{swagger_css_url}">
-    <link rel="shortcut icon" href="{swagger_favicon_url}">
-    <title>{title}</title>
+    <link type="text/css" rel="stylesheet" href="{html_module.escape(swagger_css_url, quote=True)}">
+    <link rel="shortcut icon" href="{html_module.escape(swagger_favicon_url, quote=True)}">
+    <title>{html_module.escape(title)}</title>
     </head>
     <body>
     <div id="swagger-ui">
     </div>
-    <script src="{swagger_js_url}"></script>
+    <script src="{html_module.escape(swagger_js_url, quote=True)}"></script>
     <!-- `SwaggerUIBundle` is now available on the page -->
     <script>
     const ui = SwaggerUIBundle({{
-        url: '{openapi_url}',
+        url: {json.dumps(openapi_url)},
     """

@@ -137,7 +138,7 @@ def get_swagger_ui_html(
         html += f"{json.dumps(key)}: {json.dumps(jsonable_encoder(value))},\n"

     if oauth2_redirect_url:
-        html += f"oauth2RedirectUrl: window.location.origin + '{oauth2_redirect_url}',"
+        html += f"oauth2RedirectUrl: window.location.origin + {json.dumps(oauth2_redirect_url)},"

     html += """
     presets: [
@@ -219,10 +220,10 @@ def get_redoc_html(
     html = f"""
     <!DOCTYPE html>
     <html>
     <head>
-    <title>{title}</title>
+    <title>{html_module.escape(title)}</title>
     <!-- needed for adaptive design -->
     <meta charset="utf-8"/>
     <meta name="viewport" content="width=device-width, initial-scale=1">
     """
     if with_google_fonts:
@@ -230,9 +231,9 @@ def get_redoc_html(
     <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
     """
     html += f"""
-    <link rel="shortcut icon" href="{redoc_favicon_url}">
+    <link rel="shortcut icon" href="{html_module.escape(redoc_favicon_url, quote=True)}">
     <!--
     ReDoc doesn't change outer page styles
     -->
@@ -245,8 +246,8 @@ def get_redoc_html(
     <body>
     <noscript>
         ReDoc requires Javascript to function. Please enable it to browse the documentation.
     </noscript>
-    <redoc spec-url="{openapi_url}"></redoc>
-    <script src="{redoc_js_url}"> </script>
+    <redoc spec-url="{html_module.escape(openapi_url, quote=True)}"></redoc>
+    <script src="{html_module.escape(redoc_js_url, quote=True)}"> </script>
     </body>
     </html>
     """
```

The key changes:
1. Import Python's built-in `html` module for HTML escaping
2. Use `html.escape()` for all values inserted into HTML context
3. Use `json.dumps()` for all values inserted into JavaScript context (instead of manual string quoting)
4. Use `quote=True` parameter for URL attributes to escape quotes in addition to special HTML characters