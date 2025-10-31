# Bug Report: fastapi.openapi HTML Injection in swagger_ui_parameters

**Target**: `fastapi.openapi.docs.get_swagger_ui_html`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_swagger_ui_html` function in `fastapi.openapi.docs` is vulnerable to HTML/JavaScript injection through the `swagger_ui_parameters` parameter. When parameter values contain `</script>` tags, they can break out of the enclosing `<script>` tag even though the values are JSON-encoded. This is because browsers parse HTML before executing JavaScript, so `</script>` within a JSON string still closes the script tag in HTML context.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env')

from hypothesis import given, strategies as st, settings
from fastapi.openapi.docs import get_swagger_ui_html

@given(st.dictionaries(
    st.text(min_size=1, max_size=20),
    st.text(),
    min_size=1,
    max_size=5
))
@settings(max_examples=500)
def test_swagger_ui_parameters_no_script_tag_injection(params):
    html = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Test",
        swagger_ui_parameters=params
    )

    html_str = html.body.decode()

    for value in params.values():
        if "</script>" in str(value):
            assert "<\\/script>" in html_str or "&lt;/script&gt;" in html_str, \
                   f"</script> tag not properly escaped in: {value}"
```

**Failing input**: `params={"description": "</script><script>alert('XSS')</script><script>"}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env')

from fastapi.openapi.docs import get_swagger_ui_html

test_params = {"description": "</script><script>alert('XSS')</script><script>"}

html = get_swagger_ui_html(
    openapi_url="/openapi.json",
    title="Test API",
    swagger_ui_parameters=test_params
)

html_str = html.body.decode()
start = html_str.find("description")
end = html_str.find("presets", start)
print(html_str[start:end])
```

Output:
```
description": "</script><script>alert('XSS')</script><script>",


```

The browser will parse this as:
1. First script tag contains: `SwaggerUIBundle({ url: '/openapi.json', ... "description": "`
2. Second script tag (injected) contains: ``
3. Third script tag (injected) contains: `alert('XSS')`
4. Fourth script tag (injected) contains: ``
5. Remainder: `", ... })`

## Why This Is A Bug

The vulnerable code is in `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/fastapi/openapi/docs.py` at lines 135-136:

```python
for key, value in current_swagger_ui_parameters.items():
    html += f"{json.dumps(key)}: {json.dumps(jsonable_encoder(value))},\n"
```

While `json.dumps()` properly escapes the values for JSON context (escaping quotes, backslashes, etc.), it does NOT escape `</script>` because that's not a special sequence in JSON. However, when this JSON is embedded inside an HTML `<script>` tag, the browser's HTML parser will still treat `</script>` as closing the script tag, even though it's inside a string literal.

This is a well-known XSS vulnerability pattern when embedding JSON in HTML. The OWASP recommendation is to escape `<` as `\u003c` or replace `</script>` with `<\/script>` when embedding JSON in HTML script tags.

This vulnerability could allow:
- Execution of arbitrary JavaScript through malicious configuration
- XSS attacks if user input influences swagger_ui_parameters
- Breaking the Swagger UI functionality by closing the script prematurely

## Fix

```diff
--- a/fastapi/openapi/docs.py
+++ b/fastapi/openapi/docs.py
@@ -133,7 +133,8 @@ def get_swagger_ui_html(
     """

     for key, value in current_swagger_ui_parameters.items():
-        html += f"{json.dumps(key)}: {json.dumps(jsonable_encoder(value))},\n"
+        value_json = json.dumps(jsonable_encoder(value)).replace("</", r"<\/")
+        html += f"{json.dumps(key)}: {value_json},\n"

     if oauth2_redirect_url:
         html += f"oauth2RedirectUrl: window.location.origin + '{oauth2_redirect_url}',"
```

The fix replaces `</` with `<\/` in the JSON output, which is valid JavaScript (the backslash before `/` is allowed in JavaScript strings) but prevents the HTML parser from treating `</script>` as a tag closure. This is a standard technique for safely embedding JSON in HTML script tags.