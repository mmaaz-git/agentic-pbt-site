"""Minimal reproduction of pydantic plugin build_wrapper exception masking bug."""

from unittest.mock import Mock
from pydantic.plugin._schema_validator import build_wrapper
from pydantic_core import ValidationError


def original_func(x):
    """Function that raises a ValidationError."""
    raise ValidationError.from_exception_data('validation_failed', [])


# Create two plugin handlers
handler1 = Mock()
handler1.on_error = Mock(side_effect=RuntimeError("handler1 crashed"))
handler1.on_error.__module__ = 'plugin1'

handler2 = Mock()
handler2.on_error = Mock()
handler2.on_error.__module__ = 'plugin2'

# Wrap the function with the handlers
wrapped = build_wrapper(original_func, [handler1, handler2])

# Execute and observe the behavior
try:
    wrapped(42)
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")
    print(f"Expected exception type: ValidationError")
    print(f"Actual exception type: {type(e).__name__}")
    print(f"Handler2 called: {handler2.on_error.called}")
    print(f"\nThis demonstrates the bug:")
    print(f"1. The original ValidationError is masked by handler1's RuntimeError")
    print(f"2. Handler2 never gets executed because handler1 crashed")