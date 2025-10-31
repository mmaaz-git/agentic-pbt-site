# Bug Report: Cython.Plex.Lexicon InvalidToken Constructor Mismatch

**Target**: `Cython.Plex.Lexicons.parse_token_definition`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Lexicon.parse_token_definition` method raises `InvalidToken` exceptions with only one argument (message), but the `InvalidToken` constructor requires two arguments (token_number, message), causing a TypeError instead of the intended exception.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Plex import Lexicon, Str, TEXT
from Cython.Plex.Errors import InvalidToken

@given(st.lists(st.tuples(st.just(Str('a')), st.just(TEXT)), min_size=0, max_size=10))
@settings(max_examples=200)
def test_lexicon_validation_raises_correct_exception(specs):
    invalid_specs = [
        "not a tuple",
        (Str('a'),),
        ("not an RE", TEXT)
    ]

    for spec in invalid_specs:
        try:
            lex = Lexicon([spec])
        except InvalidToken:
            pass
        except TypeError as e:
            raise AssertionError(f"Expected InvalidToken but got TypeError: {e}")
```

**Failing input**: `[(Str('a'),)]` or `[("not an RE", TEXT)]`

## Reproducing the Bug

```python
from Cython.Plex import Lexicon, Str, TEXT

try:
    Lexicon([(Str('a'),)])
except TypeError as e:
    print(f"Got TypeError: {e}")
    print("Expected: InvalidToken exception")

try:
    Lexicon([("not an RE", TEXT)])
except TypeError as e:
    print(f"Got TypeError: {e}")
    print("Expected: InvalidToken exception")
```

## Why This Is A Bug

The `InvalidToken` class in `Errors.py` line 21-22 defines `__init__(self, token_number, message)` requiring two arguments. However, `Lexicons.py` calls it with only one argument in three places:

- Line 168: `raise Errors.InvalidToken("Token definition is not a tuple")`
- Line 170: `raise Errors.InvalidToken("Wrong number of items in token definition")`
- Line 174: `raise Errors.InvalidToken("Pattern is not an RE instance")`

This causes a TypeError to be raised instead of the intended InvalidToken exception, breaking the error handling contract and making debugging more difficult.

## Fix

```diff
--- a/Cython/Plex/Lexicons.py
+++ b/Cython/Plex/Lexicons.py
@@ -165,13 +165,13 @@ class Lexicon:

     def parse_token_definition(self, token_spec):
         if not isinstance(token_spec, tuple):
-            raise Errors.InvalidToken("Token definition is not a tuple")
+            raise Errors.InvalidToken(0, "Token definition is not a tuple")
         if len(token_spec) != 2:
-            raise Errors.InvalidToken("Wrong number of items in token definition")
+            raise Errors.InvalidToken(0, "Wrong number of items in token definition")

         pattern, action = token_spec
         if not isinstance(pattern, Regexps.RE):
-            raise Errors.InvalidToken("Pattern is not an RE instance")
+            raise Errors.InvalidToken(0, "Pattern is not an RE instance")
         return (pattern, action)

     def get_initial_state(self, name):
```