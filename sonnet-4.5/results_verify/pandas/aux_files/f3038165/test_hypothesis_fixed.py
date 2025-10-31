#!/usr/bin/env python3
import pandas.errors


def test_abstract_method_error_invalid_types(invalid_methodtype):
    class DummyClass:
        pass

    instance = DummyClass()

    try:
        pandas.errors.AbstractMethodError(instance, methodtype=invalid_methodtype)
    except ValueError as e:
        error_message = str(e)

        print(f"Testing with methodtype='{invalid_methodtype}'")
        print(f"Error message: {error_message}")

        # Check all assertions from the bug report
        has_must_be = 'methodtype must be one of' in error_message
        has_invalid_type = invalid_methodtype in error_message
        valid_types = {'method', 'classmethod', 'staticmethod', 'property'}
        all_valid_in_message = all(valid_type in error_message for valid_type in valid_types)

        print(f"  'methodtype must be one of' in message: {has_must_be}")
        print(f"  '{invalid_methodtype}' in message: {has_invalid_type}")
        print(f"  All valid types in message: {all_valid_in_message}")

        # Check the specific issue - the swapped variables
        if f"methodtype must be one of {invalid_methodtype}" in error_message:
            print(f"  ERROR: Variables are swapped! Message starts with 'methodtype must be one of {invalid_methodtype}'")
            return False
        else:
            print(f"  OK: Message format appears correct")
            return True


# Test with several invalid values
test_values = ['invalid_type', 'foo', 'bar', '123', '']
for val in test_values:
    result = test_abstract_method_error_invalid_types(val)
    print()