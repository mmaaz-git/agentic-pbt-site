# Bug Report: Cython.Plex.Lexicons InvalidToken Exception Raised with Wrong Arguments

**Target**: `Cython.Plex.Lexicons.parse_token_definition`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_token_definition` method raises `InvalidToken` exceptions with only one argument instead of the required two arguments, causing a `TypeError` that masks the actual validation error messages for malformed token specifications.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Plex import Lexicon, Str
from Cython.Plex.Errors import InvalidToken
import pytest
import traceback

def test_lexicon_validation_errors():
    """Test that InvalidToken exceptions are raised properly for malformed token specs"""

    print("Testing InvalidToken exception handling in Lexicon...")
    print("="*60)

    # Test 1: Single-element tuple
    print("\nTest 1: Single-element tuple (wrong number of items)")
    try:
        with pytest.raises(InvalidToken):
            Lexicon([(Str('a'),)])
        print("ERROR: Expected TypeError but pytest.raises succeeded")
    except TypeError as e:
        print(f"FAILED: Got TypeError instead of InvalidToken")
        print(f"  TypeError message: {e}")
        print(f"  This happens because InvalidToken is raised with wrong arguments")
        traceback.print_exc()

    print("\n" + "-"*60)

    # Test 2: Non-RE pattern
    print("\nTest 2: Non-RE pattern (string instead of RE)")
    try:
        with pytest.raises(InvalidToken):
            Lexicon([("not an RE", "TEXT")])
        print("ERROR: Expected TypeError but pytest.raises succeeded")
    except TypeError as e:
        print(f"FAILED: Got TypeError instead of InvalidToken")
        print(f"  TypeError message: {e}")
        print(f"  This happens because InvalidToken is raised with wrong arguments")
        traceback.print_exc()

    print("\n" + "="*60)
    print("\nConclusion: The bug is confirmed. InvalidToken exceptions are raised")
    print("with only 1 argument (message) instead of the required 2 arguments")
    print("(token_number, message), causing TypeError instead of proper validation errors.")

if __name__ == "__main__":
    test_lexicon_validation_errors()
```

<details>

<summary>
**Failing input**: `[(Str('a'),)]` and `[("not an RE", "TEXT")]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 17, in test_lexicon_validation_errors
    Lexicon([(Str('a'),)])
    ~~~~~~~^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Lexicons.py", line 127, in __init__
    self.add_token_to_machine(
    ~~~~~~~~~~~~~~~~~~~~~~~~~^
        nfa, default_initial_state, spec, token_number)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Lexicons.py", line 149, in add_token_to_machine
    (re, action_spec) = self.parse_token_definition(token_spec)
                        ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Lexicons.py", line 170, in parse_token_definition
    raise Errors.InvalidToken("Wrong number of items in token definition")
          ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: InvalidToken.__init__() missing 1 required positional argument: 'message'
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/11/hypo.py", line 31, in test_lexicon_validation_errors
    Lexicon([("not an RE", "TEXT")])
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Lexicons.py", line 127, in __init__
    self.add_token_to_machine(
    ~~~~~~~~~~~~~~~~~~~~~~~~~^
        nfa, default_initial_state, spec, token_number)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Lexicons.py", line 149, in add_token_to_machine
    (re, action_spec) = self.parse_token_definition(token_spec)
                        ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Lexicons.py", line 174, in parse_token_definition
    raise Errors.InvalidToken("Pattern is not an RE instance")
          ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: InvalidToken.__init__() missing 1 required positional argument: 'message'
Testing InvalidToken exception handling in Lexicon...
============================================================

Test 1: Single-element tuple (wrong number of items)
FAILED: Got TypeError instead of InvalidToken
  TypeError message: InvalidToken.__init__() missing 1 required positional argument: 'message'
  This happens because InvalidToken is raised with wrong arguments

------------------------------------------------------------

Test 2: Non-RE pattern (string instead of RE)
FAILED: Got TypeError instead of InvalidToken
  TypeError message: InvalidToken.__init__() missing 1 required positional argument: 'message'
  This happens because InvalidToken is raised with wrong arguments

============================================================

Conclusion: The bug is confirmed. InvalidToken exceptions are raised
with only 1 argument (message) instead of the required 2 arguments
(token_number, message), causing TypeError instead of proper validation errors.
```
</details>

## Reproducing the Bug

```python
from Cython.Plex import Lexicon, Str

# Test case 1: Single-element tuple (wrong number of items)
print("Test 1: Single-element tuple")
try:
    Lexicon([(Str('a'),)])
except TypeError as e:
    print(f"Got TypeError: {e}")
    print("Expected: InvalidToken with message 'Wrong number of items in token definition'")
except Exception as e:
    print(f"Got {type(e).__name__}: {e}")

print("\n" + "="*60 + "\n")

# Test case 2: Non-RE pattern
print("Test 2: Non-RE pattern (string instead of RE)")
try:
    Lexicon([("not an RE", "TEXT")])
except TypeError as e:
    print(f"Got TypeError: {e}")
    print("Expected: InvalidToken with message 'Pattern is not an RE instance'")
except Exception as e:
    print(f"Got {type(e).__name__}: {e}")

print("\n" + "="*60 + "\n")

# Test case 3: Non-tuple token spec (this one should work correctly)
print("Test 3: Non-tuple token spec")
try:
    Lexicon(["not a tuple"])
except Exception as e:
    print(f"Got {type(e).__name__}: {e}")
    print("This one works correctly because it uses a different code path")

print("\n" + "="*60 + "\n")

# Test case 4: Valid input (should work)
print("Test 4: Valid input")
try:
    lexicon = Lexicon([(Str('a'), "TEXT")])
    print("Success: Lexicon created successfully")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
```

<details>

<summary>
TypeError occurs instead of InvalidToken for malformed token specifications
</summary>
```
Test 1: Single-element tuple
Got TypeError: InvalidToken.__init__() missing 1 required positional argument: 'message'
Expected: InvalidToken with message 'Wrong number of items in token definition'

============================================================

Test 2: Non-RE pattern (string instead of RE)
Got TypeError: InvalidToken.__init__() missing 1 required positional argument: 'message'
Expected: InvalidToken with message 'Pattern is not an RE instance'

============================================================

Test 3: Non-tuple token spec
Got InvalidToken: Token number 1: Expected a token definition (tuple) or State instance
This one works correctly because it uses a different code path

============================================================

Test 4: Valid input
Success: Lexicon created successfully
```
</details>

## Why This Is A Bug

The `InvalidToken` exception class in `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Plex/Errors.py:20-22` requires exactly two arguments in its constructor:

```python
class InvalidToken(PlexError):
    def __init__(self, token_number, message):
        PlexError.__init__(self, "Token number %d: %s" % (token_number, message))
```

However, in `parse_token_definition` method (Lexicons.py lines 168, 170, 174), the exception is raised with only one argument:

```python
raise Errors.InvalidToken("Token definition is not a tuple")  # Line 168
raise Errors.InvalidToken("Wrong number of items in token definition")  # Line 170
raise Errors.InvalidToken("Pattern is not an RE instance")  # Line 174
```

This violates the exception's constructor contract, resulting in a `TypeError` about missing positional arguments instead of providing users with helpful validation error messages. Users receive a confusing error about the exception's internal implementation rather than clear feedback about what's wrong with their token specification.

## Relevant Context

The bug occurs in the validation path when users provide malformed token specifications to the Lexicon constructor. The same file (Lexicons.py) demonstrates correct usage of InvalidToken at lines 131-133:

```python
raise Errors.InvalidToken(
    token_number,
    "Expected a token definition (tuple) or State instance")
```

This shows that the token_number is available in the calling context (`add_token_to_machine` at line 149) but is not passed to `parse_token_definition`. The method signature needs to accept token_number as a parameter and use it when raising InvalidToken exceptions.

## Proposed Fix

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
@@ -163,14 +163,14 @@ class Lexicon:
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