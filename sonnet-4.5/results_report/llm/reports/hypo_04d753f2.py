from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import not_nulls

@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_not_nulls_removes_none_values(data):
    result = not_nulls(data)
    assert isinstance(result, dict)
    assert all(value is not None for value in result.values())

if __name__ == "__main__":
    test_not_nulls_removes_none_values()