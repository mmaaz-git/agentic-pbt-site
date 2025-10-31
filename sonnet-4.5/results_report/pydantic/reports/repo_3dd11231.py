import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from unittest.mock import Mock
from pydantic.plugin._schema_validator import build_wrapper


def func():
    raise ValueError("Original error")


handler = Mock()

def bad_on_exception(exception):
    raise RuntimeError("Handler raised an exception")

handler.on_exception = bad_on_exception

wrapper = build_wrapper(func, [handler])

try:
    wrapper()
except RuntimeError as e:
    print(f"BUG: Caught RuntimeError from handler: {e}")
    print("The original ValueError was suppressed!")
except ValueError as e:
    print(f"Expected: Caught original ValueError: {e}")