import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

import pytest
from hypothesis import given, settings, strategies as st
from unittest.mock import Mock
from pydantic.plugin._schema_validator import filter_handlers


@given(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(),
        st.lists(st.integers()),
        st.dictionaries(st.text(), st.integers()),
        st.none(),
        st.booleans(),
    )
)
@settings(max_examples=500)
def test_filter_handlers_with_objects_without_module(obj):
    handler = Mock()
    setattr(handler, 'test_method', obj)

    try:
        result = filter_handlers(handler, 'test_method')
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
    except AttributeError:
        if hasattr(obj, '__module__'):
            pytest.fail(f"filter_handlers raised AttributeError for object with __module__: {type(obj)}")
        else:
            pytest.fail(f"BUG: filter_handlers crashes on objects without __module__ attribute (type: {type(obj).__name__})")

if __name__ == "__main__":
    test_filter_handlers_with_objects_without_module()