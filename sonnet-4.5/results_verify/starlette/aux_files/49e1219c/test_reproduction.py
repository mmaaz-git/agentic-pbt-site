"""Test to reproduce the Jinja2Templates bug with empty directory sequences."""

import jinja2
from starlette.templating import Jinja2Templates
from hypothesis import given, strategies as st

# First, let's test the Hypothesis property-based test from the bug report
@given(st.sampled_from([[], (), ""]))
def test_jinja2templates_empty_directory_with_env(empty_directory):
    """Property-based test from the bug report."""
    custom_env = jinja2.Environment()

    templates = Jinja2Templates(directory=empty_directory, env=custom_env)

    assert templates.env is custom_env, \
        f"Expected templates.env to be custom_env when directory={empty_directory!r}, but got a different env"

# Now let's test the direct reproduction case from the bug report
def test_direct_reproduction():
    """Direct reproduction test from the bug report."""
    custom_env = jinja2.Environment()

    templates = Jinja2Templates(directory=[], env=custom_env)

    print(f"templates.env is custom_env: {templates.env is custom_env}")

    assert templates.env is custom_env, "Expected custom_env to be used, but a new env was created instead"


if __name__ == "__main__":
    # Run the direct test first
    print("Running direct reproduction test...")
    try:
        test_direct_reproduction()
        print("✓ Direct reproduction test passed")
    except AssertionError as e:
        print(f"✗ Direct reproduction test failed: {e}")
    except Exception as e:
        print(f"✗ Unexpected error in direct test: {e}")

    # Run the property-based test
    print("\nRunning property-based test...")
    try:
        test_jinja2templates_empty_directory_with_env()
        print("✓ Property-based test passed")
    except AssertionError as e:
        print(f"✗ Property-based test failed: {e}")
    except Exception as e:
        print(f"✗ Unexpected error in property test: {e}")