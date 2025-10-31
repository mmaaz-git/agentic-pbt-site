#!/usr/bin/env python3
"""
Property-based test for llm.default_plugins.openai_models.not_nulls
"""
from hypothesis import given, strategies as st

def not_nulls(data) -> dict:
    """
    This is the buggy implementation from llm/default_plugins/openai_models.py:915
    """
    return {key: value for key, value in data if value is not None}

@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_not_nulls_filters_none_values(d):
    result = not_nulls(d)

    assert isinstance(result, dict)
    for key, value in result.items():
        assert value is not None

if __name__ == "__main__":
    # Run the test - it will fail and show the minimal failing example
    test_not_nulls_filters_none_values()