# Bug Report: fastapi.openapi XSS Vulnerabilities in HTML Documentation Functions

**Target**: `fastapi.openapi.docs.get_swagger_ui_html` and `fastapi.openapi.docs.get_redoc_html`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_swagger_ui_html` and `get_redoc_html` functions in `fastapi.openapi.docs` do not properly escape user-controlled input parameters (`title` and `openapi_url`), allowing for Cross-Site Scripting (XSS) attacks. Attackers can inject arbitrary JavaScript code that executes in the browser when users visit the API documentation pages.

## Property-Based Test

```python
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
```

**Failing input**: `title='<'` (or any string containing HTML special characters)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/path/to/fastapi')

from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

xss_title = '<script>alert("XSS")</script>'
response = get_swagger_ui_html(openapi_url="/openapi.json", title=xss_title)
content = response.body.decode('utf-8')

assert xss_title in content

xss_url = '"><script>alert("XSS")</script>'
response = get_swagger_ui_html(openapi_url=xss_url, title="Test")
content = response.body.decode('utf-8')

assert 'alert("XSS")' in content

response = get_redoc_html(openapi_url="/openapi.json", title=xss_title)
content = response.body.decode('utf-8')

assert xss_title in content
```

## Why This Is A Bug

1. **XSS Vulnerability**: The functions directly interpolate user-controlled input into HTML using f-strings without escaping. This allows injection of arbitrary HTML and JavaScript.

2. **Attack Vectors**:
   - In `get_swagger_ui_html`: The `title` parameter is inserted into `<title>{title}</title>` without escaping
   - In `get_swagger_ui_html`: The `openapi_url` parameter is inserted into `url: '{openapi_url}'` without escaping
   - In `get_redoc_html`: The `title` parameter is inserted into `<title>{title}</title>` without escaping
   - In `get_redoc_html`: The `openapi_url` parameter is inserted into `<redoc spec-url="{openapi_url}"></redoc>` without escaping

3. **Real-World Impact**: While FastAPI applications typically hardcode these values, applications that allow user customization of documentation titles or URLs (e.g., through configuration files, environment variables, or admin panels) would be vulnerable to stored XSS.

## Fix

Escape all user-controlled input using `html.escape()` before inserting into HTML:

```diff
diff --git a/fastapi/openapi/docs.py b/fastapi/openapi/docs.py
index 1a2b3c4..5e6f7g8 100644
--- a/fastapi/openapi/docs.py
+++ b/fastapi/openapi/docs.py
@@ -1,4 +1,5 @@
 import json
+import html
 from typing import Any, Dict, Optional

 from fastapi.encoders import jsonable_encoder
@@ -117,11 +118,12 @@ def get_swagger_ui_html(
     html = f"""
     <!DOCTYPE html>
     <html>
     <head>
     <link type="text/css" rel="stylesheet" href="{swagger_css_url}">
     <link rel="shortcut icon" href="{swagger_favicon_url}">
-    <title>{title}</title>
+    <title>{html.escape(title)}</title>
     </head>
     <body>
     <div id="swagger-ui">
     </div>
@@ -129,7 +131,7 @@ def get_swagger_ui_html(
     <!-- `SwaggerUIBundle` is now available on the page -->
     <script>
     const ui = SwaggerUIBundle({{
-        url: '{openapi_url}',
+        url: '{html.escape(openapi_url, quote=True)}',
     """

     for key, value in current_swagger_ui_parameters.items():
@@ -219,7 +221,7 @@ def get_redoc_html(
     html = f"""
     <!DOCTYPE html>
     <html>
     <head>
-    <title>{title}</title>
+    <title>{html.escape(title)}</title>
     <!-- needed for adaptive design -->
     <meta charset="utf-8"/>
     <meta name="viewport" content="width=device-width, initial-scale=1">
@@ -245,7 +247,7 @@ def get_redoc_html(
     <noscript>
         ReDoc requires Javascript to function. Please enable it to browse the documentation.
     </noscript>
-    <redoc spec-url="{openapi_url}"></redoc>
+    <redoc spec-url="{html.escape(openapi_url, quote=True)}"></redoc>
     <script src="{redoc_js_url}"> </script>
     </body>
     </html>
```

Note: Additional parameters like `swagger_css_url`, `swagger_js_url`, `redoc_js_url`, etc., should also be validated or escaped if they can be controlled by users.