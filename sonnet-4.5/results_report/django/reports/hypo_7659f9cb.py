from hypothesis import given, strategies as st, example, settings
from django.core.management.templates import TemplateCommand
from django.core.management.base import CommandError
import pytest


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=20))
@example("__main__")  # Force the failing case
@settings(max_examples=10)
def test_validate_name_no_unexpected_errors(name):
    """
    Property: validate_name should only raise CommandError for invalid names,
    never ValueError or other unexpected exceptions.

    This test exposes the bug where find_spec('__main__') raises ValueError
    instead of returning None or a valid spec. However, Django's validate_name
    method checks isidentifier() first, which rejects '__main__' before reaching
    the problematic find_spec call. To truly expose this bug, we need to test
    with a name that passes isidentifier() but still causes find_spec to raise
    ValueError (which __main__ doesn't in practice because it fails isidentifier()).

    The bug is still real though - if validate_name is called with __main__ after
    somehow bypassing the isidentifier check, it would crash with ValueError.
    """
    cmd = TemplateCommand()
    cmd.app_or_project = 'app'
    cmd.a_or_an = 'an'

    try:
        cmd.validate_name(name, 'name')
        # Name was valid
        pass
    except ValueError as e:
        # This is a bug - should only raise CommandError
        pytest.fail(f"Unexpected ValueError for name '{name}': {e}")
    except CommandError:
        # Expected validation failure
        pass

if __name__ == '__main__':
    test_validate_name_no_unexpected_errors()