from hypothesis import given, settings, strategies as st
from llm.utils import remove_dict_none_values

@st.composite
def dict_with_nested_nones(draw):
    return {
        "direct_none": None,
        "direct_empty_dict": {"nested_none": None},
        "list_with_none": [None, 1, 2],
        "list_with_empty_dict": [{"nested_none": None}],
    }

@settings(max_examples=100)
@given(dict_with_nested_nones())
def test_remove_dict_none_values_consistency(d):
    result = remove_dict_none_values(d)

    def has_empty_dict(obj, path=""):
        if isinstance(obj, dict):
            if not obj:
                return True, path
            for k, v in obj.items():
                found, p = has_empty_dict(v, f"{path}.{k}")
                if found:
                    return True, p
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                found, p = has_empty_dict(item, f"{path}[{i}]")
                if found:
                    return True, p
        return False, ""

    found, path = has_empty_dict(result)
    assert not found, f"Empty dict found at {path} after remove_dict_none_values"

if __name__ == "__main__":
    test_remove_dict_none_values_consistency()