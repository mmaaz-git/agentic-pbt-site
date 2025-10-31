from pandas.util._decorators import deprecate


# Test 1: Simple reproduction test
def bad_alternative():
    """
    Summary line
    Next line immediately (no blank line after summary)
    """
    pass


print("Test 1: Basic reproduction")
try:
    result = deprecate("old_func", bad_alternative, "1.0.0")
    print("BUG: Malformed docstring was accepted!")
except AssertionError as e:
    print(f"Correctly rejected with error: {e}")

print("\n" + "="*50 + "\n")

# Test 2: Test correct format (should pass)
def good_alternative():
    """
    Summary line

    Detailed description with blank line after summary
    """
    pass

print("Test 2: Correct format")
try:
    result = deprecate("old_func", good_alternative, "1.0.0")
    print("Correct format accepted as expected")
except AssertionError as e:
    print(f"Unexpectedly rejected: {e}")

print("\n" + "="*50 + "\n")

# Test 3: Test various edge cases
test_cases = [
    ("no_blank_after_summary", """
    Summary
    Next line
    """, False),  # Should fail but doesn't

    ("no_initial_blank", """Summary
    Next line
    """, True),  # Should fail and does

    ("no_summary", """


    Details only
    """, True),  # Should fail and does

    ("correct_format", """
    Summary

    Details
    """, False),  # Should pass and does
]

print("Test 3: Edge cases")
for name, docstring, should_fail in test_cases:
    def test_func():
        pass
    test_func.__doc__ = docstring

    try:
        result = deprecate("old", test_func, "1.0")
        if should_fail:
            print(f"  {name}: BUG - Should have failed but passed")
        else:
            print(f"  {name}: OK - Passed as expected")
    except AssertionError:
        if should_fail:
            print(f"  {name}: OK - Failed as expected")
        else:
            print(f"  {name}: ERROR - Should have passed but failed")