import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from unittest.mock import Mock
from hypothesis import given, strategies as st, settings
from pydantic.plugin._schema_validator import build_wrapper


@given(st.text(), st.text())
@settings(max_examples=20)
def test_on_exception_handler_exception_suppresses_original(original_error_msg, handler_error_msg):
    def func():
        raise ValueError(original_error_msg)

    handler = Mock()

    def bad_on_exception(exception):
        raise RuntimeError(handler_error_msg)

    handler.on_exception = bad_on_exception

    wrapper = build_wrapper(func, [handler])

    try:
        wrapper()
        assert False, "Should have raised an exception"
    except RuntimeError as e:
        raise AssertionError(f"Original exception was suppressed by handler exception")
    except ValueError as e:
        pass

# Run the test
test_on_exception_handler_exception_suppresses_original()
print("Test completed")