import pandas.errors as errors


class DummyClass:
    pass


def test_with_invalid_type(invalid_methodtype):
    try:
        errors.AbstractMethodError(DummyClass, methodtype=invalid_methodtype)
        print(f"ERROR: No exception raised for '{invalid_methodtype}'")
    except ValueError as exc_info:
        error_msg = str(exc_info)
        print(f"Test with '{invalid_methodtype}':")
        print(f"  Error message: {error_msg}")

        if f"methodtype must be one of {invalid_methodtype}" in error_msg:
            print(f"  BUG CONFIRMED: Error message incorrectly says 'methodtype must be one of {invalid_methodtype}'")
            return False
        else:
            print(f"  Test passed - error message does not contain the bug pattern")
            return True

# Test with various invalid values
test_cases = ["", "invalid_type", "foo", "bar", "1234"]
for test_case in test_cases:
    test_with_invalid_type(test_case)