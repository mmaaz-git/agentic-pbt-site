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