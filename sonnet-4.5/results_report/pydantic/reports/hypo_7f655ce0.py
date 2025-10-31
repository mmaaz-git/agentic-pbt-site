"""Property-based test demonstrating pydantic plugin build_wrapper exception masking bug."""

from unittest.mock import Mock
from hypothesis import given, strategies as st
from pydantic.plugin._schema_validator import build_wrapper
from pydantic_core import ValidationError


@given(input_value=st.integers())
def test_first_on_error_exception_prevents_second_handler(input_value):
    """Test that when first on_error handler raises, second handler never executes."""
    handler1_called = []
    handler2_called = []

    def original_func(x):
        raise ValidationError.from_exception_data('test', [])

    handler1 = Mock()
    handler1.on_error = Mock(side_effect=RuntimeError("handler1 error"))
    handler1.on_error.__module__ = 'test1'

    handler2 = Mock()
    handler2.on_error = lambda e: handler2_called.append(True)
    handler2.on_error.__module__ = 'test2'

    wrapped = build_wrapper(original_func, [handler1, handler2])

    caught_exception = None
    try:
        wrapped(input_value)
    except Exception as e:
        caught_exception = e

    # Bug: The RuntimeError from handler1 masks the original ValidationError
    assert isinstance(caught_exception, RuntimeError)
    # Bug: handler2 never gets called because handler1 crashed
    assert len(handler2_called) == 0


if __name__ == "__main__":
    # Run the property-based test
    test_first_on_error_exception_prevents_second_handler()