# Bug Report: fastapi.openapi.docs Multiple XSS Vulnerabilities

**Target**: `fastapi.openapi.docs`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_swagger_ui_html` and `get_redoc_html` functions contain multiple XSS vulnerabilities where user-provided parameters are inserted directly into HTML/JavaScript without proper escaping, allowing arbitrary code execution.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.openapi.docs import get_swagger_ui_html

@given(
    title=st.text(min_size=1, max_size=100),
    openapi_url=st.text(min_size=1, max_size=200),
    oauth2_redirect_url=st.one_of(st.none(), st.text(min_size=1, max_size=200))
)
def test_no_xss_injection(title, openapi_url, oauth2_redirect_url):
    """HTML/JS should be properly escaped to prevent XSS"""
    html = get_swagger_ui_html(
        openapi_url=openapi_url,
        title=title,
        oauth2_redirect_url=oauth2_redirect_url
    )
    html_content = html.body.decode('utf-8')

    assert "</script>" not in html_content or "&lt;/script&gt;" in html_content
    assert "javascript:" not in openapi_url or "'" not in html_content
```

**Failing inputs**: `title="</title><script>alert(1)</script>"`, `openapi_url="javascript:alert(1)"`, `oauth2_redirect_url="'/><script>alert(1)</script>"`

## Reproducing the Bug

```python
from fastapi.openapi.docs import get_swagger_ui_html

html = get_swagger_ui_html(
    openapi_url="/openapi.json",
    title="</title><script>alert('XSS')</script>",
    oauth2_redirect_url="'/><script>alert('OAuth2 XSS')</script>"
)

content = html.body.decode('utf-8')
print("<script>alert" in content)
```

Output: `True` - The malicious scripts are injected unescaped into the HTML.

## Why This Is A Bug

Multiple XSS vulnerabilities exist in `docs.py` where f-strings insert user input without escaping:

1. **Line 123**: `<title>{title}</title>` - allows breaking out of title tag
2. **Line 132**: `url: '{openapi_url}',` - allows JavaScript injection
3. **Line 139**: `oauth2RedirectUrl: window.location.origin + '{oauth2_redirect_url}',` - allows JS injection
4. **Line 223**: `<title>{title}</title>` (ReDoc) - same vulnerability
5. **Line 248**: `<redoc spec-url="{openapi_url}"></redoc>` - allows attribute escape

XSS can lead to session hijacking, credential theft, malicious actions, and data exfiltration.

## Fix

Use `html.escape()` for HTML context and `json.dumps()` for JavaScript context:

```diff
--- a/fastapi/openapi/docs.py
+++ b/fastapi/openapi/docs.py
@@ -1,5 +1,6 @@
 import json
 from typing import Any, Dict, Optional
+import html

 from fastapi.encoders import jsonable_encoder
 from starlette.responses import HTMLResponse
@@ -120,7 +121,7 @@
     <head>
     <link type="text/css" rel="stylesheet" href="{swagger_css_url}">
     <link rel="shortcut icon" href="{swagger_favicon_url}">
-    <title>{title}</title>
+    <title>{html.escape(title)}</title>
     </head>
     <body>
     <div id="swagger-ui">
@@ -129,7 +130,7 @@
     <!-- `SwaggerUIBundle` is now available on the page -->
     <script>
     const ui = SwaggerUIBundle({{
-        url: '{openapi_url}',
+        url: {json.dumps(openapi_url)},
     """

     for key, value in current_swagger_ui_parameters.items():
@@ -137,7 +138,7 @@

     if oauth2_redirect_url:
-        html += f"oauth2RedirectUrl: window.location.origin + '{oauth2_redirect_url}',"
+        html += f"oauth2RedirectUrl: window.location.origin + {json.dumps(oauth2_redirect_url)},"

     html += """
     presets: [
@@ -220,7 +221,7 @@
     <!DOCTYPE html>
     <html>
     <head>
-    <title>{title}</title>
+    <title>{html.escape(title)}</title>
     <!-- needed for adaptive design -->
     <meta charset="utf-8"/>
     <meta name="viewport" content="width=device-width, initial-scale=1">
@@ -245,7 +246,7 @@
     <noscript>
         ReDoc requires Javascript to function. Please enable it to browse the documentation.
     </noscript>
-    <redoc spec-url="{openapi_url}"></redoc>
+    <redoc spec-url={json.dumps(openapi_url)}></redoc>
     <script src="{redoc_js_url}"> </script>
     </body>
     </html>
```