# Bug Report: Cython.Build.Dependencies.parse_list KeyError on Unclosed Quotes

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_list` function crashes with a `KeyError` when processing strings containing unclosed quotes due to incorrect string slicing that removes the trailing underscore from internally-generated label placeholders.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import parse_list


@settings(max_examples=500)
@given(st.text())
def test_parse_list_returns_list(s):
    result = parse_list(s)
    assert isinstance(result, list), f"parse_list should return a list, got {type(result)}"


if __name__ == "__main__":
    test_parse_list_returns_list()
```

<details>

<summary>
**Failing input**: `"'"`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 13, in <module>
    test_parse_list_returns_list()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 6, in test_parse_list_returns_list
    @given(st.text())
                   ^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 8, in test_parse_list_returns_list
    result = parse_list(s)
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 135, in parse_list
    return [unquote(item) for item in s.split(delimiter) if item.strip()]
            ~~~~~~~^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py", line 132, in unquote
    return literals[literal[1:-1]]
           ~~~~~~~~^^^^^^^^^^^^^^^
KeyError: '__Pyx_L1'
Falsifying example: test_parse_list_returns_list(
    s="'",
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py:132
        /home/npc/miniconda/lib/python3.13/site-packages/Cython/Build/Dependencies.py:352
```
</details>

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

# Test case 1: Single double-quote character
print("Test 1: Single double-quote character")
try:
    result = parse_list('"')
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
print()

# Test case 2: Single quote character
print("Test 2: Single quote character")
try:
    result = parse_list("'")
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
print()

# Test case 3: Two double-quotes
print("Test 3: Two double-quotes")
try:
    result = parse_list('""')
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
print()

# Test case 4: Two single quotes
print("Test 4: Two single quotes")
try:
    result = parse_list("''")
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
```

<details>

<summary>
KeyError crashes on unclosed quotes
</summary>
```
Test 1: Single double-quote character
KeyError: '__Pyx_L1'

Test 2: Single quote character
KeyError: '__Pyx_L1'

Test 3: Two double-quotes
KeyError: ''

Test 4: Two single quotes
KeyError: ''
```
</details>

## Why This Is A Bug

This violates expected behavior because the function crashes with an internal implementation-detail error (`KeyError: '__Pyx_L1'`) rather than handling malformed input gracefully. The issue stems from a subtle interaction between two functions:

1. **`strip_string_literals()` behavior**: When encountering unclosed quotes, this function (at line 309) correctly handles the edge case by creating a label with format `"__Pyx_L{counter}_"` (note the trailing underscore) and storing the unclosed string fragment in a dictionary with this label as the key.

2. **`parse_list()` incorrect assumption**: The inner `unquote()` function (at line 132) incorrectly assumes all quoted strings are complete and strips both first and last characters using `literal[1:-1]`. For labels like `"__Pyx_L1_"`, this removes the trailing underscore, resulting in `"__Pyx_L1"` which doesn't exist in the literals dictionary.

The documentation doesn't specify expected behavior for malformed input, but crashing with an obscure `KeyError` that exposes internal implementation details is clearly unintentional and provides no useful feedback to users about what went wrong.

## Relevant Context

The `parse_list` function is an internal utility in Cython's build system used for parsing configuration values and dependencies. It's located in `/Cython/Build/Dependencies.py` and is designed to parse space-delimited or comma-delimited lists with support for quoted strings.

The function includes a comment at line 308 acknowledging unclosed quotes: `"# This probably indicates an unclosed string literal, i.e. a broken file."` However, the current implementation doesn't handle this case gracefully.

Documentation link: The function is not part of Cython's public API documentation.
Code location: [Cython/Build/Dependencies.py:110-135](https://github.com/cython/cython/blob/master/Cython/Build/Dependencies.py#L110-L135)

## Proposed Fix

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -128,7 +128,10 @@ def parse_list(s):
     s, literals = strip_string_literals(s)
     def unquote(literal):
         literal = literal.strip()
-        if literal[0] in "'\"":
+        if not literal:
+            return literal
+        # Check if this is a properly quoted string (has both opening and closing quotes)
+        if literal[0] in "'\"" and len(literal) > 1 and literal[-1] in "'\"":
             return literals[literal[1:-1]]
         else:
             return literal
```