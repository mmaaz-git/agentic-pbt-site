# Bug Report: Flask select_jinja_autoescape Case Sensitivity Security Issue

**Target**: `Flask.select_jinja_autoescape`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Flask's `select_jinja_autoescape` method performs case-sensitive file extension matching, which causes autoescaping to be incorrectly disabled for uppercase or mixed-case HTML/XML file extensions (e.g., `.HTML`, `.XML`). This is a **security vulnerability** as it can lead to XSS attacks on Windows systems where file extensions are case-insensitive.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from flask import Flask

app = Flask(__name__)

@given(st.sampled_from(['.html', '.htm', '.xml', '.xhtml', '.svg']))
def test_case_insensitive_autoescape(ext):
    """
    Property: Autoescaping should be enabled for HTML/XML files
    regardless of extension case.
    """
    lowercase_filename = f"template{ext}"
    uppercase_filename = f"template{ext.upper()}"

    lowercase_result = app.select_jinja_autoescape(lowercase_filename)
    uppercase_result = app.select_jinja_autoescape(uppercase_filename)

    assert lowercase_result is True
    assert uppercase_result is True, \
        f"Case sensitivity bug: {uppercase_filename} should also return True"
```

**Failing input**: `.html` (and all other extensions: `.htm`, `.xml`, `.xhtml`, `.svg`)

## Reproducing the Bug

```python
from flask import Flask

app = Flask(__name__)

print(app.select_jinja_autoescape('template.html'))
print(app.select_jinja_autoescape('template.HTML'))
print(app.select_jinja_autoescape('template.xml'))
print(app.select_jinja_autoescape('template.XML'))
```

Output:
```
True
False
True
False
```

## Why This Is A Bug

1. **Windows file systems are case-insensitive**: On Windows, `template.html` and `template.HTML` refer to the same file. A user might save a file as `Template.HTML` and expect autoescaping to work.

2. **Security implications**: Autoescaping protects against XSS attacks. If a user creates a template with an uppercase extension, they would reasonably expect autoescaping to be enabled (since the file is HTML), but Flask would silently disable it, creating a security vulnerability.

3. **Violates principle of least surprise**: The docstring for `select_jinja_autoescape` says it "Returns `True` if autoescaping should be active for the given template name" without mentioning case sensitivity. Users would not expect `.HTML` to behave differently from `.html`.

4. **Inconsistent with common practice**: Most web frameworks and template engines treat file extensions as case-insensitive for security-related features.

## Fix

```diff
def select_jinja_autoescape(self, filename: str) -> bool:
    """Returns ``True`` if autoescaping should be active for the given
    template name. If no template name is given, returns `True`.

    .. versionchanged:: 2.2
        Autoescaping is now enabled by default for ``.svg`` files.

    .. versionadded:: 0.5
    """
    if filename is None:
        return True
-   return filename.endswith((".html", ".htm", ".xml", ".xhtml", ".svg"))
+   return filename.lower().endswith((".html", ".htm", ".xml", ".xhtml", ".svg"))
```

This simple fix converts the filename to lowercase before checking the extension, making the comparison case-insensitive and preventing the security issue.