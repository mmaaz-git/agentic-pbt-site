import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from unittest.mock import Mock
from pydantic.plugin._schema_validator import build_wrapper
from pydantic_core import ValidationError


def func():
    raise ValidationError.from_exception_data("Original validation error", [])


handler = Mock()

def bad_on_error(error):
    raise RuntimeError("Handler raised an exception")

handler.on_error = bad_on_error

wrapper = build_wrapper(func, [handler])

try:
    wrapper()
except RuntimeError as e:
    print(f"BUG: Caught RuntimeError from handler: {e}")
    print("The original ValidationError was suppressed!")
except ValidationError as e:
    print(f"Expected: Caught original ValidationError: {e}")