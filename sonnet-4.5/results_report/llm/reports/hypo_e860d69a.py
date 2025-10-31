#!/usr/bin/env python3
"""Property-based test for llm.utils.remove_dict_none_values using Hypothesis"""

from llm.utils import remove_dict_none_values
from hypothesis import given, strategies as st

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

# Run the test
if __name__ == "__main__":
    try:
        test_remove_dict_none_values_removes_all_nones()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed with error: {e}")
        print("\nThis confirms the bug - the function does not remove None values from lists.")