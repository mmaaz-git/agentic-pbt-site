# Bug Report: Cython.Plex.Lexicons.parse_token_definition Incorrect InvalidToken Usage

**Target**: `Cython.Plex.Lexicons.parse_token_definition`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_token_definition` method raises `InvalidToken` exceptions with incorrect arguments, causing `TypeError` instead of the intended validation error when users provide malformed token specifications.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Plex import Lexicon, Str
from Cython.Plex.Errors import InvalidToken
import pytest

def test_lexicon_validation_errors():
    with pytest.raises(InvalidToken):
        Lexicon([(Str('a'),)])

    with pytest.raises(InvalidToken):
        Lexicon([("not an RE", "TEXT")])
```

**Failing input**: Any malformed token specification (wrong tuple length or non-RE pattern)

## Reproducing the Bug

```python
from Cython.Plex import Lexicon, Str

try:
    Lexicon([(Str('a'),)])
except TypeError as e:
    print(f"Got TypeError: {e}")
    print("Expected: InvalidToken with helpful message")
```

Output:
```
Got TypeError: InvalidToken.__init__() missing 1 required positional argument: 'message'
Expected: InvalidToken with helpful message
```

## Why This Is A Bug

The `InvalidToken` exception class requires two arguments: `token_number` and `message` (see Errors.py:21-22). However, `parse_token_definition` raises it with only one argument in three places (Lexicons.py:168, 170, 174):

```python
raise Errors.InvalidToken("Token definition is not a tuple")
raise Errors.InvalidToken("Wrong number of items in token definition")
raise Errors.InvalidToken("Pattern is not an RE instance")
```

This causes a `TypeError` instead of providing users with a helpful error message about their malformed token specification. The error obscures the actual problem (malformed token spec) behind an implementation detail (missing constructor argument).

## Fix

```diff
--- a/Cython/Plex/Lexicons.py
+++ b/Cython/Plex/Lexicons.py
@@ -146,7 +146,7 @@ class Lexicon:

     def add_token_to_machine(self, machine, initial_state, token_spec, token_number):
         try:
-            (re, action_spec) = self.parse_token_definition(token_spec)
+            (re, action_spec) = self.parse_token_definition(token_spec, token_number)
             if isinstance(action_spec, Actions.Action):
                 action = action_spec
             else:
@@ -163,13 +163,13 @@ class Lexicon:
         except Errors.PlexError as e:
             raise e.__class__("Token number %d: %s" % (token_number, e))

-    def parse_token_definition(self, token_spec):
+    def parse_token_definition(self, token_spec, token_number):
         if not isinstance(token_spec, tuple):
-            raise Errors.InvalidToken("Token definition is not a tuple")
+            raise Errors.InvalidToken(token_number, "Token definition is not a tuple")
         if len(token_spec) != 2:
-            raise Errors.InvalidToken("Wrong number of items in token definition")
+            raise Errors.InvalidToken(token_number, "Wrong number of items in token definition")

         pattern, action = token_spec
         if not isinstance(pattern, Regexps.RE):
-            raise Errors.InvalidToken("Pattern is not an RE instance")
+            raise Errors.InvalidToken(token_number, "Pattern is not an RE instance")
         return (pattern, action)
```