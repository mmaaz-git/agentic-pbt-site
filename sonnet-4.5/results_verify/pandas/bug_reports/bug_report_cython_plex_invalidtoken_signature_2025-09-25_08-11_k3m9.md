# Bug Report: Cython.Plex InvalidToken Raised with Wrong Signature

**Target**: `Cython.Plex.Lexicons.parse_token_definition`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_token_definition` method raises `InvalidToken` exceptions with only one argument (message), but `InvalidToken.__init__` requires two arguments (token_number, message), causing TypeError instead of proper error handling.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Plex import Lexicon, TEXT
from Cython.Plex.Errors import InvalidToken
import pytest

@given(st.text(min_size=1, max_size=10))
def test_lexicon_rejects_non_re_pattern(pattern_str):
    with pytest.raises(InvalidToken):
        Lexicon([(pattern_str, TEXT)])
```

**Failing input**: Any string (e.g., `'0'`)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, 'lib/python3.13/site-packages')

from Cython.Plex import Lexicon, TEXT

try:
    Lexicon([("not_a_regex", TEXT)])
except TypeError as e:
    print(f"Got TypeError instead of InvalidToken: {e}")
except Exception as e:
    print(f"Got {type(e).__name__}: {e}")
```

Output:
```
Got TypeError instead of InvalidToken: InvalidToken.__init__() missing 1 required positional argument: 'message'
```

## Why This Is A Bug

The `InvalidToken` exception class requires two arguments (Errors.py:20-22):

```python
class InvalidToken(PlexError):
    def __init__(self, token_number, message):
        PlexError.__init__(self, "Token number %d: %s" % (token_number, message))
```

However, `parse_token_definition` raises it with only one argument (Lexicons.py:168, 170, 174):

```python
if not isinstance(token_spec, tuple):
    raise Errors.InvalidToken("Token definition is not a tuple")  # Missing token_number!
if len(token_spec) != 2:
    raise Errors.InvalidToken("Wrong number of items in token definition")  # Missing token_number!
# ...
if not isinstance(pattern, Regexps.RE):
    raise Errors.InvalidToken("Pattern is not an RE instance")  # Missing token_number!
```

This causes:
1. TypeError when constructing the exception
2. The TypeError isn't caught by the `except Errors.PlexError` handler (line 163)
3. Users get confusing TypeError instead of informative InvalidToken
4. Token number information is lost from error messages

## Fix

Pass the token_number argument to InvalidToken in parse_token_definition. Since parse_token_definition doesn't have access to token_number, it should be passed as a parameter:

```diff
--- a/Cython/Plex/Lexicons.py
+++ b/Cython/Plex/Lexicons.py
@@ -145,9 +145,9 @@ class Lexicon:
         self.machine = dfa

     def add_token_to_machine(self, machine, initial_state, token_spec, token_number):
         try:
-            (re, action_spec) = self.parse_token_definition(token_spec)
+            (re, action_spec) = self.parse_token_definition(token_spec, token_number)
             if isinstance(action_spec, Actions.Action):
                 action = action_spec
             else:
@@ -164,14 +164,14 @@ class Lexicon:
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