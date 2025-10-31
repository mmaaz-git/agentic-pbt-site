import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from unittest.mock import Mock
from hypothesis import given, strategies as st, settings
from pydantic.plugin._schema_validator import build_wrapper
from pydantic_core import ValidationError


@given(st.text(), st.text())
@settings(max_examples=20)
def test_on_error_handler_exception_suppresses_original(original_error_msg, handler_error_msg):
    def func():
        raise ValidationError.from_exception_data(original_error_msg, [])

    handler = Mock()

    def bad_on_error(error):
        raise RuntimeError(handler_error_msg)

    handler.on_error = bad_on_error

    wrapper = build_wrapper(func, [handler])

    try:
        wrapper()
        assert False, "Should have raised an exception"
    except RuntimeError as e:
        raise AssertionError(f"Original ValidationError was suppressed by handler exception")
    except ValidationError as e:
        pass

if __name__ == "__main__":
    test_on_error_handler_exception_suppresses_original()