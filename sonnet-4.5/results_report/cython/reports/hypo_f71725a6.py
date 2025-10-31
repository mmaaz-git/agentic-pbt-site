import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
import string

@given(st.text(alphabet=string.ascii_letters + '_', min_size=1, max_size=10).filter(str.isidentifier))
@settings(max_examples=100)
def test_fill_command_py_prefix_strips_prefix(var_name):
    """Test that py: prefix is correctly stripped from variable names in fill_command"""
    arg_string = f"py:{var_name}"

    # Simulate the buggy parsing logic from Cython.Tempita.fill_command
    name = arg_string
    if name.startswith('py:'):
        parsed_name = name[:3]  # BUG: This keeps 'py:' instead of removing it

    expected_name = var_name
    actual_name = parsed_name

    assert actual_name == expected_name, f"Variable name should be {expected_name!r}, got {actual_name!r}"

if __name__ == "__main__":
    # Run the test to find a failing example
    print("Running Hypothesis property-based test to find failing inputs...")
    print("=" * 60)
    try:
        test_fill_command_py_prefix_strips_prefix()
        print("Test passed (no bug found)")
    except AssertionError as e:
        print(f"Test failed as expected, demonstrating the bug!")
        print(f"Error: {e}")