import os
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/flask_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
from flask.helpers import get_debug_flag, get_load_dotenv

@given(val=st.sampled_from(["0", "false", "no"]))
def test_debug_flag_should_handle_whitespace(val):
    original = os.environ.get("FLASK_DEBUG")
    try:
        os.environ["FLASK_DEBUG"] = f" {val} "
        result = get_debug_flag()
        assert result is False, f"Should be False for ' {val} ' (with spaces)"
    finally:
        if original is None:
            os.environ.pop("FLASK_DEBUG", None)
        else:
            os.environ["FLASK_DEBUG"] = original

@given(val=st.sampled_from(["0", "false", "no"]))
def test_load_dotenv_should_handle_whitespace(val):
    original = os.environ.get("FLASK_SKIP_DOTENV")
    try:
        os.environ["FLASK_SKIP_DOTENV"] = f" {val} "
        result = get_load_dotenv()
        assert result is True, f"Should be True for ' {val} ' (with spaces)"
    finally:
        if original is None:
            os.environ.pop("FLASK_SKIP_DOTENV", None)
        else:
            os.environ["FLASK_SKIP_DOTENV"] = original

# Run tests
print("Running test_debug_flag_should_handle_whitespace...")
test_debug_flag_should_handle_whitespace()

print("\nRunning test_load_dotenv_should_handle_whitespace...")
test_load_dotenv_should_handle_whitespace()