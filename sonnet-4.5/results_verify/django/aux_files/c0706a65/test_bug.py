#!/usr/bin/env python3
"""Test script to reproduce the reported bug"""

from django.core.checks.registry import CheckRegistry
from hypothesis import given, strategies as st


# First, test the reproduction code from the bug report
def test_basic_reproduction():
    print("=" * 60)
    print("Testing basic reproduction from bug report")
    print("=" * 60)

    registry = CheckRegistry()

    def bad_check(app_configs, **kwargs):
        return "error message"

    registry.register(bad_check, "test")

    result = registry.run_checks()

    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
    print(f"Length of result: {len(result)}")
    print(f"Expected: list of CheckMessage objects")
    print(f"Actual: {result}")
    print(f"Result equals list('error message'): {result == list('error message')}")
    print()
    return result


# Test the property-based test from the bug report
@given(st.text(min_size=1))
def test_registry_check_returns_string_bug(error_string):
    registry = CheckRegistry()

    def bad_check(app_configs, **kwargs):
        return error_string

    registry.register(bad_check, "test")

    result = registry.run_checks()

    assert len(result) == len(error_string)
    assert result == list(error_string)


# Test what happens with a proper check function
def test_correct_usage():
    print("=" * 60)
    print("Testing correct usage with CheckMessage objects")
    print("=" * 60)

    from django.core.checks.messages import Error, Warning

    registry = CheckRegistry()

    def good_check(app_configs, **kwargs):
        return [
            Error("This is an error", hint="Fix this error"),
            Warning("This is a warning", hint="Consider this")
        ]

    registry.register(good_check, "test")

    result = registry.run_checks()

    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
    print(f"Length of result: {len(result)}")
    print(f"First item type: {type(result[0]) if result else 'No items'}")
    print()
    return result


# Test with empty string
def test_empty_string():
    print("=" * 60)
    print("Testing with empty string")
    print("=" * 60)

    registry = CheckRegistry()

    def bad_check(app_configs, **kwargs):
        return ""

    registry.register(bad_check, "test")

    result = registry.run_checks()

    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
    print(f"Length of result: {len(result)}")
    print()
    return result


# Test with other iterables
def test_other_iterables():
    print("=" * 60)
    print("Testing with other iterables")
    print("=" * 60)

    # Test with tuple
    registry1 = CheckRegistry()
    def tuple_check(app_configs, **kwargs):
        return ("a", "b", "c")
    registry1.register(tuple_check, "test")
    result1 = registry1.run_checks()
    print(f"Tuple result: {result1}")

    # Test with generator
    registry2 = CheckRegistry()
    def gen_check(app_configs, **kwargs):
        return (x for x in ["x", "y", "z"])
    registry2.register(gen_check, "test")
    result2 = registry2.run_checks()
    print(f"Generator result: {result2}")

    # Test with set
    registry3 = CheckRegistry()
    def set_check(app_configs, **kwargs):
        return {"item1", "item2"}
    registry3.register(set_check, "test")
    result3 = registry3.run_checks()
    print(f"Set result (order may vary): {sorted(result3)}")
    print()


# Test what the error message says
def test_error_message_content():
    print("=" * 60)
    print("Testing the actual error message from Django")
    print("=" * 60)

    # Look at the error message in the code
    print("Error message in code at line 92-93:")
    print('"The function %r did not return a list. All functions "')
    print('"registered with the checks registry must return a list."')
    print()
    print("Note: The error says 'must return a list' but actually")
    print("checks for Iterable, which includes strings!")
    print()


if __name__ == "__main__":
    # Run all tests
    basic_result = test_basic_reproduction()
    test_correct_usage()
    test_empty_string()
    test_other_iterables()
    test_error_message_content()

    # Run hypothesis test
    print("=" * 60)
    print("Running property-based test with Hypothesis")
    print("=" * 60)
    try:
        test_registry_check_returns_string_bug()
        print("Hypothesis test passed - confirmed bug exists!")
    except AssertionError as e:
        print(f"Hypothesis test failed unexpectedly: {e}")