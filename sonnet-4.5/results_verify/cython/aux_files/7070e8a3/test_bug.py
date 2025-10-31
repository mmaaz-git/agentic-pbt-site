#!/usr/bin/env python3
"""Test script to reproduce the reported bug"""

import sys
import traceback

# Test 1: Check if the bug actually occurs
print("Testing the reported bug...")
try:
    from Cython.Plex import Lexicon, Str

    # This should raise an InvalidToken error, but instead raises TypeError
    result = Lexicon([(Str('a'),)])
    print("ERROR: No exception raised!")
except TypeError as e:
    print(f"TypeError caught (as reported): {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Other exception caught: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test 2: Check another case
print("Testing with a non-RE pattern...")
try:
    from Cython.Plex import Lexicon

    result = Lexicon([("not an RE", "TEXT")])
    print("ERROR: No exception raised!")
except TypeError as e:
    print(f"TypeError caught: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Other exception caught: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*50 + "\n")

# Test 3: Verify the InvalidToken class signature
print("Checking InvalidToken class signature...")
from Cython.Plex.Errors import InvalidToken
import inspect

sig = inspect.signature(InvalidToken.__init__)
print(f"InvalidToken.__init__ signature: {sig}")
print(f"Parameters: {list(sig.parameters.keys())}")

# Test creating InvalidToken properly
try:
    err = InvalidToken(1, "Test message")
    print(f"Created InvalidToken successfully: {err}")
except Exception as e:
    print(f"Failed to create InvalidToken: {e}")

# Test creating InvalidToken with wrong arguments (as used in the buggy code)
try:
    err = InvalidToken("Test message")
    print(f"Created InvalidToken with single arg (SHOULD FAIL): {err}")
except TypeError as e:
    print(f"TypeError as expected when using single arg: {e}")