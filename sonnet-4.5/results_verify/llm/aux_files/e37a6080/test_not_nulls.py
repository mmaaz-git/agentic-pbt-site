from hypothesis import given, strategies as st

# First, let's create the buggy version as reported
def not_nulls_buggy(data) -> dict:
    return {key: value for key, value in data if value is not None}

# The fixed version
def not_nulls_fixed(data) -> dict:
    return {key: value for key, value in data.items() if value is not None}

# Hypothesis test
@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_not_nulls_filters_none_values(d):
    try:
        result = not_nulls_buggy(d)
        assert isinstance(result, dict)
        for key, value in result.items():
            assert value is not None
        print(f"Buggy version succeeded with: {d}")
    except ValueError as e:
        print(f"Buggy version failed with {d}: {e}")

    # Test fixed version
    result_fixed = not_nulls_fixed(d)
    assert isinstance(result_fixed, dict)
    for key, value in result_fixed.items():
        assert value is not None
    print(f"Fixed version succeeded with: {d}")

# Manual reproduction test
def test_manual():
    print("\n=== Manual test with specific input ===")
    test_data = {'temperature': 0.7, 'max_tokens': None, 'top_p': 0.9}

    print(f"Input: {test_data}")
    try:
        result = not_nulls_buggy(test_data)
        print(f"Buggy version result: {result}")
    except ValueError as e:
        print(f"Buggy version error: {e}")

    result_fixed = not_nulls_fixed(test_data)
    print(f"Fixed version result: {result_fixed}")

    # Test with the simple failing input
    print("\n=== Test with {'a': 1} ===")
    simple_test = {'a': 1}
    try:
        result = not_nulls_buggy(simple_test)
        print(f"Buggy version result: {result}")
    except ValueError as e:
        print(f"Buggy version error: {e}")

if __name__ == "__main__":
    # Run manual tests
    test_manual()

    # Run property-based test
    print("\n=== Running property-based test ===")
    test_not_nulls_filters_none_values()