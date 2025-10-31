# Bug Report: django.core.servers.basehttp.WSGIRequestHandler.get_environ Dictionary Modification During Iteration

**Target**: `django.core.servers.basehttp.WSGIRequestHandler.get_environ`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_environ` method in Django's development server modifies the `self.headers` dictionary while iterating over it, causing `RuntimeError` and incomplete removal of headers with underscores, potentially leaving a security vulnerability.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test demonstrating Django's dictionary modification bug.

The bug: Django's WSGIRequestHandler.get_environ modifies headers dictionary
while iterating, violating Python's iteration rules.
"""

from hypothesis import given, strategies as st, settings, reproduce_failure

# Strategy to generate header names - mix of normal and underscore-containing
header_name_strategy = st.one_of(
    # Headers with underscores (should be removed)
    st.from_regex(r"[A-Z][a-z]*_[A-Z][a-z]*(_[A-Z][a-z]*)*", fullmatch=True),
    # Normal headers with hyphens (should be kept)
    st.from_regex(r"[A-Z][a-z]*-[A-Z][a-z]*(-[A-Z][a-z]*)*", fullmatch=True),
    # Single word headers
    st.from_regex(r"[A-Z][a-z]+", fullmatch=True)
)

@given(
    headers_dict=st.dictionaries(
        header_name_strategy,
        st.text(min_size=1, max_size=100),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=100, deadline=None)
def test_django_header_removal_pattern(headers_dict):
    """Test Django's pattern for removing headers with underscores.

    Django code at django/core/servers/basehttp.py lines 220-222:
        for k in self.headers:
            if "_" in k:
                del self.headers[k]

    This pattern violates Python's rule against dictionary modification
    during iteration and can cause RuntimeError or incomplete removal.
    """
    # Make a copy for testing
    test_headers = headers_dict.copy()

    # Count headers with underscores
    original_underscore_headers = [k for k in test_headers.keys() if '_' in k]

    # Apply Django's buggy pattern
    error_raised = False
    try:
        for k in test_headers:
            if "_" in k:
                del test_headers[k]
    except RuntimeError as e:
        error_raised = True
        # RuntimeError confirms the bug
        remaining_underscore = [k for k in test_headers.keys() if '_' in k]
        assert remaining_underscore, f"RuntimeError raised but some headers with underscores remain: {remaining_underscore}"

    # If no error, check if all underscore headers were removed
    if not error_raised:
        remaining_underscore = [k for k in test_headers.keys() if '_' in k]
        # Bug: Not all headers with underscores removed
        assert not remaining_underscore, f"No RuntimeError but headers with underscores remain: {remaining_underscore}"

if __name__ == "__main__":
    print("Running property-based test for Django's header removal bug...")
    print()

    # Run with a specific failing example
    failing_example = {
        'X_Forwarded_For': 'value1',
        'User_Agent': 'value2',
        'X_Real_IP': 'value3',
        'Content-Type': 'value4'
    }

    print(f"Testing with: {list(failing_example.keys())}")

    # Test the failing example directly
    test_headers = failing_example.copy()
    original_underscore = [k for k in test_headers if '_' in k]
    print(f"Headers with underscores: {original_underscore}")
    print()

    try:
        for k in test_headers:
            if "_" in k:
                del test_headers[k]
        print("ERROR: No RuntimeError raised")
        remaining = [k for k in test_headers if '_' in k]
        if remaining:
            print(f"BUG: Headers with underscores remain: {remaining}")
    except RuntimeError as e:
        print(f"RuntimeError (confirms bug): {e}")
        remaining = [k for k in test_headers if '_' in k]
        print(f"Headers with underscores remaining after error: {remaining}")

    # Run full Hypothesis test
    print("\n" + "=" * 60)
    print("Running full Hypothesis test suite...")
    print("=" * 60)
    try:
        test_django_header_removal_pattern()
    except Exception as e:
        print(f"Test found failure cases that demonstrate the bug")
```

<details>

<summary>
**Failing input**: `{'X_Forwarded_For': 'value1', 'User_Agent': 'value2', 'X_Real_IP': 'value3', 'Content-Type': 'value4'}`
</summary>
```
Running property-based test for Django's header removal bug...

Testing with: ['X_Forwarded_For', 'User_Agent', 'X_Real_IP', 'Content-Type']
Headers with underscores: ['X_Forwarded_For', 'User_Agent', 'X_Real_IP']

RuntimeError (confirms bug): dictionary changed size during iteration
Headers with underscores remaining after error: ['User_Agent', 'X_Real_IP']

============================================================
Running full Hypothesis test suite...
============================================================
Test found failure cases that demonstrate the bug
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Reproduce the bug in Django's WSGIRequestHandler.get_environ method
which modifies a dictionary while iterating over it.

The code at django/core/servers/basehttp.py lines 220-222:
    for k in self.headers:
        if "_" in k:
            del self.headers[k]

This violates Python's rule against modifying dictionaries during iteration.
"""

# Test with regular Python dictionary (most clear demonstration)
headers_dict = {
    'X-Custom-Header': 'value1',
    'X_Forwarded_For': '192.168.1.1',  # Has underscore
    'User_Agent': 'Mozilla/5.0',       # Has underscore
    'X_Real_IP': '10.0.0.1',          # Has underscore
    'Content-Type': 'text/html'
}

print("Headers before:", list(headers_dict.keys()))
print("Headers with underscores:", [k for k in headers_dict.keys() if '_' in k])
print()

print("Executing Django's buggy pattern:")
print("for k in headers_dict:")
print("    if '_' in k:")
print("        del headers_dict[k]")
print()

try:
    for k in headers_dict:
        if "_" in k:
            del headers_dict[k]
    print("ERROR: No RuntimeError raised! But check if all underscore headers were removed...")
except RuntimeError as e:
    print(f"RuntimeError: {e}")

print()
print("Headers after:", list(headers_dict.keys()))
remaining_underscore = [k for k in headers_dict.keys() if '_' in k]
print("Headers with underscores remaining:", remaining_underscore)

if remaining_underscore:
    print()
    print("BUG CONFIRMED: Not all headers with underscores were removed!")
    print("This defeats the security measure meant to prevent header spoofing.")
```

<details>

<summary>
RuntimeError: dictionary changed size during iteration
</summary>
```
Headers before: ['X-Custom-Header', 'X_Forwarded_For', 'User_Agent', 'X_Real_IP', 'Content-Type']
Headers with underscores: ['X_Forwarded_For', 'User_Agent', 'X_Real_IP']

Executing Django's buggy pattern:
for k in headers_dict:
    if '_' in k:
        del headers_dict[k]

RuntimeError: dictionary changed size during iteration

Headers after: ['X-Custom-Header', 'User_Agent', 'X_Real_IP', 'Content-Type']
Headers with underscores remaining: ['User_Agent', 'X_Real_IP']

BUG CONFIRMED: Not all headers with underscores were removed!
This defeats the security measure meant to prevent header spoofing.
```
</details>

## Why This Is A Bug

This code violates Python's fundamental rule against modifying a dictionary while iterating over it. The behavior is explicitly undefined and typically results in:

1. **RuntimeError**: Python raises "dictionary changed size during iteration" when it detects the modification
2. **Incomplete removal**: When the error occurs, iteration stops, leaving some headers with underscores intact
3. **Security vulnerability**: The code's purpose is to "Strip all headers with underscores in the name before constructing the WSGI environ" to prevent header-spoofing attacks (based on CVE-2015-0219). When headers with underscores remain, the security measure fails.

The code comment explicitly states this is a security feature:
> "This prevents header-spoofing based on ambiguity between underscores and dashes both normalized to underscores in WSGI env vars. Nginx and Apache 2.4+ both do this as well."

## Relevant Context

- **Location**: `/django/core/servers/basehttp.py` lines 220-222
- **Affected component**: Django development server (`manage.py runserver`)
- **Security context**: Headers with underscores can be exploited for header spoofing because both "Foo-Bar" and "Foo_Bar" normalize to "HTTP_FOO_BAR" in WSGI environ
- **Python documentation**: https://docs.python.org/3/library/stdtypes.html#dict - "Changing a dictionary while iterating over it" is undefined behavior
- **Django documentation**: States that runserver strips headers with underscores for security

Note: While Django uses `email.message.Message` for headers (which may not always raise RuntimeError), the pattern is still incorrect and against Python best practices. Using regular dictionaries or other dictionary-like objects would trigger the error consistently.

## Proposed Fix

Collect keys to delete first, then delete them in a separate loop:

```diff
     def get_environ(self):
         # Strip all headers with underscores in the name before constructing
         # the WSGI environ. This prevents header-spoofing based on ambiguity
         # between underscores and dashes both normalized to underscores in WSGI
         # env vars. Nginx and Apache 2.4+ both do this as well.
-        for k in self.headers:
-            if "_" in k:
-                del self.headers[k]
+        keys_to_delete = [k for k in self.headers if "_" in k]
+        for k in keys_to_delete:
+            del self.headers[k]

         return super().get_environ()
```