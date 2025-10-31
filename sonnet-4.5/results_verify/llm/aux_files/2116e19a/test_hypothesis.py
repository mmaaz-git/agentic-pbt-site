import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from llm.utils import remove_dict_none_values


def has_none_value(obj):
    """Recursively check if obj contains any None values"""
    if obj is None:
        return True
    if isinstance(obj, dict):
        for value in obj.values():
            if has_none_value(value):
                return True
    elif isinstance(obj, list):
        for item in obj:
            if has_none_value(item):
                return True
    return False


@given(st.recursive(
    st.one_of(st.none(), st.integers(), st.text(), st.booleans()),
    lambda children: st.one_of(
        st.dictionaries(st.text(), children, max_size=5),
        st.lists(children, max_size=5)
    ),
    max_leaves=10
))
@settings(max_examples=100)  # Using fewer examples for faster testing
def test_remove_dict_none_values_removes_all_none(d):
    result = remove_dict_none_values(d)
    if isinstance(result, dict):
        assert not has_none_value(result), \
            f"Found None values in output:\nInput: {d}\nOutput: {result}"

# Run the test
if __name__ == "__main__":
    try:
        test_remove_dict_none_values_removes_all_none()
        print("All tests passed")
    except AssertionError as e:
        print(f"Test failed: {e}")