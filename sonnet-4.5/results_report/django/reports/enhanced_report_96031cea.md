# Bug Report: Django HTTP parse_cookie Whitespace Key Collision Causes Data Loss

**Target**: `django.http.cookie.parse_cookie`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_cookie` function silently loses data when multiple cookies have keys consisting entirely of whitespace characters, as they all get stripped to an empty string key causing dictionary collision where only the last value survives.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, assume
from django.http.cookie import parse_cookie
import sys

# Strategy for whitespace strings that will collide when stripped
whitespace_chars = [' ', '\t', '\n', '\r', '\f', '\v']
whitespace_text = st.text(alphabet=whitespace_chars, min_size=1, max_size=5)

# Property: parse_cookie should preserve all cookies without data loss
@given(
    # Generate multiple cookies where some may have whitespace-only keys
    cookies=st.lists(
        st.tuples(
            st.one_of(
                whitespace_text,  # Whitespace-only keys
                st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789_', min_size=1, max_size=10)  # Normal keys
            ),
            st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789', min_size=1, max_size=10)  # Values
        ),
        min_size=2,
        max_size=10,
        unique_by=lambda x: x[0]  # Ensure unique keys in input
    )
)
@settings(max_examples=1000, database=None)
def test_parse_cookie_preserves_all_cookies(cookies):
    """
    This test checks if parse_cookie preserves all cookies without data loss.
    """
    # Build cookie string
    cookie_string = "; ".join(f"{k}={v}" for k, v in cookies)

    # Parse the cookie string
    parsed = parse_cookie(cookie_string)

    # Group cookies by their stripped key
    key_groups = {}
    for key, value in cookies:
        stripped_key = key.strip()
        if stripped_key not in key_groups:
            key_groups[stripped_key] = []
        key_groups[stripped_key].append((key, value))

    # Check for data loss
    for stripped_key, group in key_groups.items():
        if len(group) > 1:
            # Multiple cookies will collide to the same key after stripping
            print(f"\nFailing example found!")
            print(f"Input cookies: {cookies}")
            print(f"Cookie string: {cookie_string!r}")
            print(f"Parsed result: {parsed}")
            print(f"\nCollision detected for stripped key {stripped_key!r}:")
            print(f"  Original keys that collide: {[k for k, v in group]!r}")
            print(f"  Original values: {[v for k, v in group]!r}")
            if stripped_key in parsed:
                print(f"  Only kept value: {parsed[stripped_key]!r}")
            print(f"\nData loss: {len(group)} cookies collapsed to 1")

            # This demonstrates the bug
            assert False, f"Data loss: {len(group)} cookies with keys {[k for k,v in group]!r} all collapsed to key {stripped_key!r}"

if __name__ == "__main__":
    # Run the property test
    print("Running Hypothesis property-based test for parse_cookie...")
    print("This test verifies that parse_cookie preserves all cookies without data loss.")
    print("-" * 60)

    try:
        test_parse_cookie_preserves_all_cookies()
        print("\nAll tests passed! No data loss detected.")
    except AssertionError as e:
        print(f"\n** TEST FAILED **")
        print(f"Assertion Error: {e}")
        print("\nThe test demonstrates that parse_cookie loses data when multiple")
        print("cookies have keys that become identical after stripping whitespace.")
        sys.exit(1)
```

<details>

<summary>
**Failing input**: `[('\n', '0'), (' ', '0')]`
</summary>
```
Running Hypothesis property-based test for parse_cookie...
This test verifies that parse_cookie preserves all cookies without data loss.
------------------------------------------------------------

Failing example found!
Input cookies: [('\n', '0'), (' ', '0')]
Cookie string: '\n=0;  =0'
Parsed result: {'': '0'}

Collision detected for stripped key '':
  Original keys that collide: ['\n', ' ']
  Original values: ['0', '0']
  Only kept value: '0'

Data loss: 2 cookies collapsed to 1

Failing example found!
Input cookies: [('\n', '0rr41'), ('\r', 'owsu45a3iz'), ('g9c', 'ud1qrobc'), ('\t \r', '3hucxbf1z1')]
Cookie string: '\n=0rr41; \r=owsu45a3iz; g9c=ud1qrobc; \t \r=3hucxbf1z1'
Parsed result: {'': '3hucxbf1z1', 'g9c': 'ud1qrobc'}

Collision detected for stripped key '':
  Original keys that collide: ['\n', '\r', '\t \r']
  Original values: ['0rr41', 'owsu45a3iz', '3hucxbf1z1']
  Only kept value: '3hucxbf1z1'

Data loss: 3 cookies collapsed to 1

Failing example found!
Input cookies: [('\r', 'owsu45a3iz'), (' ', 'g9c')]
Cookie string: '\r=owsu45a3iz;  =g9c'
Parsed result: {'': 'g9c'}

Collision detected for stripped key '':
  Original keys that collide: ['\r', ' ']
  Original values: ['owsu45a3iz', 'g9c']
  Only kept value: 'g9c'

Data loss: 2 cookies collapsed to 1

Failing example found!
Input cookies: [('cyt9', 'ypguvd'), ('tyj8u', 'es'), ('\n', 'vobqgvm'), ('\x0b\x0c\r\t\r', '81br3n'), ('9_2', 'he7u2qjlmn'), ('xnbr6wsn', 'bpw4b')]
Cookie string: 'cyt9=ypguvd; tyj8u=es; \n=vobqgvm; \x0b\x0c\r\t\r=81br3n; 9_2=he7u2qjlmn; xnbr6wsn=bpw4b'
Parsed result: {'cyt9': 'ypguvd', 'tyj8u': 'es', '': '81br3n', '9_2': 'he7u2qjlmn', 'xnbr6wsn': 'bpw4b'}

Collision detected for stripped key '':
  Original keys that collide: ['\n', '\x0b\x0c\r\t\r']
  Original values: ['vobqgvm', '81br3n']
  Only kept value: '81br3n'

Data loss: 2 cookies collapsed to 1

Failing example found!
Input cookies: [('\n', 'vobqgvm'), ('tyj8u', 'es'), (' ', '0')]
Cookie string: '\n=vobqgvm; tyj8u=es;  =0'
Parsed result: {'': '0', 'tyj8u': 'es'}

Collision detected for stripped key '':
  Original keys that collide: ['\n', ' ']
  Original values: ['vobqgvm', '0']
  Only kept value: '0'

Data loss: 2 cookies collapsed to 1

Failing example found!
Input cookies: [('\n', 'vobqgvm'), ('tyj8u', 'es'), (' ', 'vobqgvm')]
Cookie string: '\n=vobqgvm; tyj8u=es;  =vobqgvm'
Parsed result: {'': 'vobqgvm', 'tyj8u': 'es'}

Collision detected for stripped key '':
  Original keys that collide: ['\n', ' ']
  Original values: ['vobqgvm', 'vobqgvm']
  Only kept value: 'vobqgvm'

Data loss: 2 cookies collapsed to 1

Failing example found!
Input cookies: [('\n', 'vobqgvm'), (' ', '0')]
Cookie string: '\n=vobqgvm;  =0'
Parsed result: {'': '0'}

Collision detected for stripped key '':
  Original keys that collide: ['\n', ' ']
  Original values: ['vobqgvm', '0']
  Only kept value: '0'

Data loss: 2 cookies collapsed to 1

Failing example found!
Input cookies: [('\n', '0'), (' ', '0')]
Cookie string: '\n=0;  =0'
Parsed result: {'': '0'}

Collision detected for stripped key '':
  Original keys that collide: ['\n', ' ']
  Original values: ['0', '0']
  Only kept value: '0'

Data loss: 2 cookies collapsed to 1

Failing example found!
Input cookies: [(' ', '0'), ('\n', '0')]
Cookie string: ' =0; \n=0'
Parsed result: {'': '0'}

Collision detected for stripped key '':
  Original keys that collide: [' ', '\n']
  Original values: ['0', '0']
  Only kept value: '0'

Data loss: 2 cookies collapsed to 1

Failing example found!
Input cookies: [(' ', '0'), ('\r', '0')]
Cookie string: ' =0; \r=0'
Parsed result: {'': '0'}

Collision detected for stripped key '':
  Original keys that collide: [' ', '\r']
  Original values: ['0', '0']
  Only kept value: '0'

Data loss: 2 cookies collapsed to 1

Failing example found!
Input cookies: [(' ', '0'), ('\r', '0')]
Cookie string: ' =0; \r=0'
Parsed result: {'': '0'}

Collision detected for stripped key '':
  Original keys that collide: [' ', '\r']
  Original values: ['0', '0']
  Only kept value: '0'

Data loss: 2 cookies collapsed to 1

** TEST FAILED **
Assertion Error: Data loss: 2 cookies with keys [' ', '\r'] all collapsed to key ''

The test demonstrates that parse_cookie loses data when multiple
cookies have keys that become identical after stripping whitespace.
```
</details>

## Reproducing the Bug

```python
from django.http.cookie import parse_cookie

# Test case 1: Multiple cookies with whitespace-only names
cookie_string = " =first; \t=second; \n=third"
result = parse_cookie(cookie_string)

print("Test case 1: Multiple cookies with whitespace-only names")
print(f"Input:  {cookie_string!r}")
print(f"Output: {result}")
print(f"Expected: 3 separate cookies")
print(f"Actual:   {len(result)} cookie(s)")
print()

# Test case 2: Mix of whitespace-only and normal cookies
cookie_string2 = "normal=value1; \t=whitespace_tab; valid=value2; \n=whitespace_newline"
result2 = parse_cookie(cookie_string2)

print("Test case 2: Mix of whitespace-only and normal cookies")
print(f"Input:  {cookie_string2!r}")
print(f"Output: {result2}")
print(f"Expected: 4 cookies (2 normal, 2 whitespace)")
print(f"Actual:   {len(result2)} cookie(s)")
print()

# Test case 3: Different whitespace characters colliding
cookie_string3 = "\r=carriage_return; \n=newline; \t=tab;  =space"
result3 = parse_cookie(cookie_string3)

print("Test case 3: Different whitespace characters all colliding")
print(f"Input:  {cookie_string3!r}")
print(f"Output: {result3}")
print(f"Expected: 4 different cookies")
print(f"Actual:   {len(result3)} cookie(s)")
print()

# Demonstrate the data loss explicitly
print("Data Loss Summary:")
print("- All whitespace-only cookie names get stripped to empty string ''")
print("- Dictionary can only hold one value per key")
print("- Only the last cookie with whitespace-only name survives")
```

<details>

<summary>
Output demonstrating data loss
</summary>
```
Test case 1: Multiple cookies with whitespace-only names
Input:  ' =first; \t=second; \n=third'
Output: {'': 'third'}
Expected: 3 separate cookies
Actual:   1 cookie(s)

Test case 2: Mix of whitespace-only and normal cookies
Input:  'normal=value1; \t=whitespace_tab; valid=value2; \n=whitespace_newline'
Output: {'normal': 'value1', '': 'whitespace_newline', 'valid': 'value2'}
Expected: 4 cookies (2 normal, 2 whitespace)
Actual:   3 cookie(s)

Test case 3: Different whitespace characters all colliding
Input:  '\r=carriage_return; \n=newline; \t=tab;  =space'
Output: {'': 'space'}
Expected: 4 different cookies
Actual:   1 cookie(s)

Data Loss Summary:
- All whitespace-only cookie names get stripped to empty string ''
- Dictionary can only hold one value per key
- Only the last cookie with whitespace-only name survives
```
</details>

## Why This Is A Bug

The `parse_cookie` function in `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/http/cookie.py` lines 19-22 contains logic that causes silent data loss:

```python
key, val = key.strip(), val.strip()
if key or val:
    cookiedict[key] = val
```

When cookie names consist entirely of whitespace characters (e.g., `\r`, `\n`, `\t`, ` `), they all get stripped to the empty string `""` on line 19. Since Python dictionaries can only store one value per key, multiple cookies with whitespace-only names overwrite each other in line 22, with only the last value surviving.

This violates expected behavior in several ways:

1. **Silent Data Loss**: The function silently discards data without any warning or error. When processing `" =first; \t=second; \n=third"`, it returns `{'': 'third'}`, losing the values `'first'` and `'second'`.

2. **RFC 6265 Violation**: According to RFC 6265 (HTTP State Management Mechanism), cookie names should be tokens that exclude whitespace characters. Whitespace-only cookie names are explicitly invalid. The function should either reject them or handle them in a way that doesn't cause data loss.

3. **Inconsistent Behavior**: The function accepts invalid input but processes it incorrectly, creating an ambiguous situation where the user cannot distinguish between different cookies that had distinct (albeit invalid) names.

4. **No Documentation**: The function's docstring simply states "Return a dictionary parsed from a `Cookie:` header string" without specifying how invalid cookies are handled or warning about potential data loss.

## Relevant Context

The parse_cookie function is used by Django's WSGI handler to parse HTTP_COOKIE headers into the request.COOKIES dictionary. This is core functionality affecting all Django applications that process cookies.

The code includes a comment referencing Mozilla Bugzilla issue #169091 (line 16-17) for handling cookies without '=' characters, showing that edge case handling has been considered, but the whitespace collision issue was overlooked.

While whitespace-only cookie names are unusual in practice and violate RFC 6265, the principle of "fail loudly rather than silently" suggests that silent data loss is worse than either rejecting invalid input or preserving it in some form.

Django documentation: https://docs.djangoproject.com/en/stable/ref/request-response/#django.http.HttpRequest.COOKIES
RFC 6265: https://datatracker.ietf.org/doc/html/rfc6265#section-4.1.1

## Proposed Fix

Skip cookies with empty keys after stripping to prevent collision:

```diff
--- a/django/http/cookie.py
+++ b/django/http/cookie.py
@@ -18,7 +18,9 @@ def parse_cookie(cookie):
             key, val = "", chunk
         key, val = key.strip(), val.strip()
-        if key or val:
+        # Skip cookies with empty keys to prevent collision
+        # (multiple whitespace-only keys would all become '')
+        if key:  # Only accept cookies with non-empty keys after stripping
             # unquote using Python's algorithm.
             cookiedict[key] = cookies._unquote(val)
     return cookiedict
```