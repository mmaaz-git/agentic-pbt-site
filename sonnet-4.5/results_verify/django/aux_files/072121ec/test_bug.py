import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.core.servers import basehttp


def test_is_broken_pipe_error_no_active_exception():
    result = basehttp.is_broken_pipe_error()
    assert result == False, "Should return False when no exception is active"

# Run the test
if __name__ == "__main__":
    try:
        test_is_broken_pipe_error_no_active_exception()
        print("Test should have failed but didn't!")
    except TypeError as e:
        print(f"Test failed with TypeError: {e}")
    except AssertionError as e:
        print(f"Test failed with AssertionError: {e}")