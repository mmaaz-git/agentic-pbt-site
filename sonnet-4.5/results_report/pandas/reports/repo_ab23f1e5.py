from pandas.util._decorators import deprecate


def bad_alternative():
    """
    Summary line
    Next line immediately (no blank line after summary)
    More content here
    """
    pass


def good_alternative():
    """
    Summary line

    Proper blank line after summary
    """
    pass


print("Testing malformed docstring (no blank line after summary)...")
try:
    result = deprecate("old_func", bad_alternative, "1.0.0")
    print("BUG: Malformed docstring was accepted!")
    print(f"Result type: {type(result)}")
except AssertionError as e:
    print("Correctly rejected with error:")
    print(str(e))

print("\nTesting properly formatted docstring...")
try:
    result = deprecate("old_func", good_alternative, "1.0.0")
    print("SUCCESS: Properly formatted docstring was accepted")
    print(f"Result type: {type(result)}")
except AssertionError as e:
    print("ERROR: Good docstring was rejected:")
    print(str(e))