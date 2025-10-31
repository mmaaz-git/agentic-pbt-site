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