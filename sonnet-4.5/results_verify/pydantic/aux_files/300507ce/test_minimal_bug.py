import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from unittest.mock import Mock
from pydantic.plugin._schema_validator import filter_handlers

handler = Mock()
handler.on_enter = 0

try:
    result = filter_handlers(handler, 'on_enter')
    print(f"Result: {result}")
except AttributeError as e:
    print(f"AttributeError: {e}")