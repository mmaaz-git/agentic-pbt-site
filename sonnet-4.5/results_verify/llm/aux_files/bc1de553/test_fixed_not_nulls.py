#!/usr/bin/env python3
"""Test the proposed fix for not_nulls function."""

def not_nulls_fixed(data) -> dict:
    """Fixed version of not_nulls with .items() added."""
    return {key: value for key, value in data.items() if value is not None}

def test_fixed_function():
    """Test that the fixed function works correctly."""

    # Test with empty dict
    result = not_nulls_fixed({})
    assert result == {}, f"Empty dict failed: {result}"
    print("✓ Empty dict works")

    # Test with None values
    result = not_nulls_fixed({'': None})
    assert result == {}, f"Dict with None value failed: {result}"
    print("✓ Dict with None value works")

    # Test with mixed values
    result = not_nulls_fixed({'key1': 'value1', 'key2': None, 'key3': 42})
    assert result == {'key1': 'value1', 'key3': 42}, f"Mixed values failed: {result}"
    print("✓ Mixed values work")

    # Test with all None values
    result = not_nulls_fixed({'a': None, 'b': None})
    assert result == {}, f"All None values failed: {result}"
    print("✓ All None values work")

    # Test with no None values
    result = not_nulls_fixed({'x': 1, 'y': 'test', 'z': []})
    assert result == {'x': 1, 'y': 'test', 'z': []}, f"No None values failed: {result}"
    print("✓ No None values work")

    # Test with falsy values (should keep them)
    result = not_nulls_fixed({'a': 0, 'b': '', 'c': False, 'd': None, 'e': []})
    assert result == {'a': 0, 'b': '', 'c': False, 'e': []}, f"Falsy values failed: {result}"
    print("✓ Falsy values handled correctly (only None removed)")

    # Run hypothesis test
    from hypothesis import given, strategies as st

    @given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
    def test_not_nulls_removes_none_values(data):
        result = not_nulls_fixed(data)
        assert isinstance(result, dict)
        assert all(value is not None for value in result.values())
        # Check all non-None values are preserved
        for key, value in data.items():
            if value is not None:
                assert key in result
                assert result[key] == value

    test_not_nulls_removes_none_values()
    print("✓ Hypothesis tests pass")

if __name__ == "__main__":
    print("Testing fixed not_nulls function...")
    test_fixed_function()
    print("\nAll tests passed! The fix works correctly.")