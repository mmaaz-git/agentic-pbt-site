# Bug Report: fastapi.openapi JavaScript Injection in oauth2_redirect_url

**Target**: `fastapi.openapi.docs.get_swagger_ui_html`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_swagger_ui_html` function in `fastapi.openapi.docs` is vulnerable to JavaScript injection through the `oauth2_redirect_url` parameter. The parameter value is directly interpolated into a JavaScript string literal without proper escaping, allowing attackers to inject arbitrary JavaScript code that will execute in users' browsers.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env')

from hypothesis import given, strategies as st, assume
from fastapi.openapi.docs import get_swagger_ui_html

@given(st.text())
def test_oauth2_redirect_url_quote_injection(url):
    assume("'" in url or '"' in url or "\n" in url)

    html = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Test",
        oauth2_redirect_url=url
    )

    html_str = html.body.decode()
    oauth_line_start = html_str.find("oauth2RedirectUrl")
    if oauth_line_start == -1:
        return

    oauth_line_end = html_str.find("\n", oauth_line_start)
    oauth_line = html_str[oauth_line_start:oauth_line_end]

    if "'" in url:
        assert "\\'" in oauth_line or "\\x" in oauth_line, \
            f"Single quote in URL not properly escaped: {repr(url)} -> {repr(oauth_line)}"
```

**Failing input**: `url="'"`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env')

from fastapi.openapi.docs import get_swagger_ui_html

malicious_url = "';alert('XSS');//"

html = get_swagger_ui_html(
    openapi_url="/openapi.json",
    title="Test API",
    oauth2_redirect_url=malicious_url
)

html_str = html.body.decode()
start_idx = html_str.find("oauth2RedirectUrl")
end_idx = html_str.find("\n", start_idx)
print(html_str[start_idx:end_idx])
```

Output:
```
oauth2RedirectUrl: window.location.origin + '';alert('XSS');//',
```

The single quote in the input terminates the JavaScript string literal, allowing `alert('XSS')` to execute, with `;//` commenting out the rest of the line.

## Why This Is A Bug

The vulnerable code is in `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/fastapi/openapi/docs.py` at line 139:

```python
if oauth2_redirect_url:
    html += f"oauth2RedirectUrl: window.location.origin + '{oauth2_redirect_url}',"
```

The `oauth2_redirect_url` value is directly inserted into a JavaScript string literal using an f-string without any escaping. This violates basic security principles for generating HTML/JavaScript content. When a user-controlled or configuration-controlled URL contains special characters like single quotes, double quotes, backslashes, or newlines, it can break out of the string context and inject arbitrary JavaScript.

This is a Cross-Site Scripting (XSS) vulnerability that could allow:
- Execution of arbitrary JavaScript in users' browsers
- Session hijacking
- Credential theft
- Defacement of the API documentation page

## Fix

```diff
--- a/fastapi/openapi/docs.py
+++ b/fastapi/openapi/docs.py
@@ -136,7 +136,7 @@ def get_swagger_ui_html(
         html += f"{json.dumps(key)}: {json.dumps(jsonable_encoder(value))},\n"

     if oauth2_redirect_url:
-        html += f"oauth2RedirectUrl: window.location.origin + '{oauth2_redirect_url}',"
+        html += f"oauth2RedirectUrl: window.location.origin + {json.dumps(oauth2_redirect_url)},"

     html += """
     presets: [
```

The fix uses `json.dumps()` to properly escape the `oauth2_redirect_url` value for JavaScript context, just like the code does for `swagger_ui_parameters` on line 136. This ensures that special characters like quotes, backslashes, and control characters are properly escaped.