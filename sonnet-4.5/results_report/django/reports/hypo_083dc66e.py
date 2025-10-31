import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from django.core.checks import CheckMessage
from django.core.checks.registry import CheckRegistry


@given(st.text(min_size=1))
def test_run_checks_should_reject_string_return_value(error_string):
    registry = CheckRegistry()

    def my_check(app_configs=None, **kwargs):
        return error_string

    registry.register(my_check)
    errors = registry.run_checks()

    for err in errors:
        assert isinstance(err, CheckMessage), (
            f"Expected CheckMessage, got {type(err).__name__}: {repr(err)}. "
            f"This suggests the check function returned a string which was "
            f"incorrectly treated as an iterable of CheckMessages."
        )

# Run the test
if __name__ == "__main__":
    test_run_checks_should_reject_string_return_value()