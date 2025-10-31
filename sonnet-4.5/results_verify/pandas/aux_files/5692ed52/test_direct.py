import pandas.errors

# Test with specific value from bug report
def test_with_methodtype(methodtype):
    """Test that invalid methodtypes raise ValueError with correct message."""
    class TestClass:
        pass

    instance = TestClass()

    try:
        pandas.errors.AbstractMethodError(instance, methodtype=methodtype)
    except ValueError as exc_info:
        error_msg = str(exc_info)

        print(f"Testing methodtype={repr(methodtype)}")
        print(f"Error message: {error_msg}")

        # Check if message has correct structure
        assert "methodtype must be one of" in error_msg

        # The bug: the message has parameters swapped
        # Check if the invalid value appears after 'got'
        if f"got {methodtype}" in error_msg or repr(methodtype) in error_msg:
            print("  ✓ Pass: Found invalid value after 'got'")
            return True
        else:
            print(f"  ✗ FAIL: Expected 'got {methodtype}' but the invalid value is in the wrong position")
            print(f"         The message incorrectly says: {error_msg}")
            return False

# Test with value from bug report
print("Test 1: methodtype='0'")
result1 = test_with_methodtype('0')

print("\nTest 2: methodtype='invalid'")
result2 = test_with_methodtype('invalid')

print("\nTest 3: methodtype='foo'")
result3 = test_with_methodtype('foo')

if not (result1 and result2 and result3):
    print("\n❌ BUG CONFIRMED: The error message has swapped parameters")