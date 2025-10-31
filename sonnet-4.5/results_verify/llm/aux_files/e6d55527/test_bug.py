#!/usr/bin/env python3
from llm.utils import remove_dict_none_values
from hypothesis import given, strategies as st

# First, the hypothesis test from the bug report
@given(st.dictionaries(
    st.text(min_size=1, max_size=10),
    st.one_of(
        st.none(),
        st.integers(),
        st.text(),
        st.lists(st.one_of(st.none(), st.integers()))
    )
))
def test_remove_dict_none_values_removes_all_nones(d):
    result = remove_dict_none_values(d)

    def has_none(obj):
        if obj is None:
            return True
        if isinstance(obj, dict):
            return any(has_none(v) for v in obj.values())
        if isinstance(obj, list):
            return any(has_none(v) for v in obj)
        return False

    assert not has_none(result), f"Result still contains None: {result}"

# Simple reproduction test
def test_simple_reproduction():
    test_dict = {"a": [1, None, 3], "b": {"c": [None, 2]}}
    result = remove_dict_none_values(test_dict)

    print(f"Input:  {test_dict}")
    print(f"Output: {result}")

    # Check if None values are in lists
    if None in result.get("a", []):
        print("BUG CONFIRMED: None values remain in list 'a'")
    if "b" in result and "c" in result["b"] and None in result["b"]["c"]:
        print("BUG CONFIRMED: None values remain in nested list 'b.c'")

    return result

if __name__ == "__main__":
    print("=== Simple Reproduction Test ===")
    result = test_simple_reproduction()
    print()

    print("=== Testing with the failing input from report ===")
    failing_input = {"a": [1, None, 3]}
    result = remove_dict_none_values(failing_input)
    print(f"Failing input: {failing_input}")
    print(f"Result: {result}")
    print(f"Does result contain None? {None in result.get('a', [])}")
    print()

    print("=== Running Hypothesis test ===")
    try:
        # Call the test directly with the test data
        d = {"a": [1, None, 3]}
        result = remove_dict_none_values(d)

        def has_none(obj):
            if obj is None:
                return True
            if isinstance(obj, dict):
                return any(has_none(v) for v in obj.values())
            if isinstance(obj, list):
                return any(has_none(v) for v in obj)
            return False

        assert not has_none(result), f"Result still contains None: {result}"
        print("Hypothesis test passed (unexpected)")
    except AssertionError as e:
        print(f"Hypothesis test failed as expected: {e}")