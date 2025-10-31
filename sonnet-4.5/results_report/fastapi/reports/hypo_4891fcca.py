from hypothesis import given, strategies as st, settings, assume
from fastapi.dependencies.utils import get_typed_annotation
import keyword


@settings(max_examples=200)
@given(
    st.sampled_from(keyword.kwlist),  # Use Python's list of keywords directly
)
def test_get_typed_annotation_handles_keywords(type_str):
    try:
        result = get_typed_annotation(type_str, {})
    except (NameError, AttributeError):
        pass
    except SyntaxError:
        assert False, f"SyntaxError should not be raised for valid identifier '{type_str}'"

# Run the test
if __name__ == "__main__":
    test_get_typed_annotation_handles_keywords()