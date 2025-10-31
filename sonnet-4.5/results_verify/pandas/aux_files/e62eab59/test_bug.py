#!/usr/bin/env python3
"""Test script to reproduce the Cython.Plex.Lexicons bug"""

from Cython.Plex import Lexicon, Str
from Cython.Plex.Errors import InvalidToken

print("=" * 60)
print("Test 1: Testing with single-element tuple (should raise InvalidToken)")
print("=" * 60)

try:
    Lexicon([(Str('a'),)])
    print("ERROR: Should have raised an exception!")
except TypeError as e:
    print(f"Got TypeError: {e}")
    print("ERROR: Expected InvalidToken with helpful message, but got TypeError instead")
except InvalidToken as e:
    print(f"Got InvalidToken as expected: {e}")

print()
print("=" * 60)
print("Test 2: Testing with non-RE pattern (should raise InvalidToken)")
print("=" * 60)

try:
    Lexicon([("not an RE", "TEXT")])
    print("ERROR: Should have raised an exception!")
except TypeError as e:
    print(f"Got TypeError: {e}")
    print("ERROR: Expected InvalidToken with helpful message, but got TypeError instead")
except InvalidToken as e:
    print(f"Got InvalidToken as expected: {e}")

print()
print("=" * 60)
print("Test 3: Testing with non-tuple token spec (should raise InvalidToken)")
print("=" * 60)

try:
    Lexicon(["not a tuple"])
    print("ERROR: Should have raised an exception!")
except TypeError as e:
    print(f"Got TypeError: {e}")
    print("ERROR: Expected InvalidToken with helpful message, but got TypeError instead")
except InvalidToken as e:
    print(f"Got InvalidToken as expected: {e}")

print()
print("=" * 60)
print("Test 4: Testing correct usage (should work fine)")
print("=" * 60)

try:
    lexicon = Lexicon([(Str('a'), "TEXT")])
    print("SUCCESS: Lexicon created successfully with valid input")
except Exception as e:
    print(f"ERROR: Unexpected exception: {e}")