import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import string
from Cython.Tempita._tempita import fill_command

@given(st.text(alphabet=string.ascii_letters + '_', min_size=1).filter(str.isidentifier))
def test_fill_command_py_prefix_removal(var_name):
    args = ['-', f'py:{var_name}=42']

    # Mock stdout to capture output
    import io
    import sys
    old_stdin = sys.stdin
    old_stdout = sys.stdout

    try:
        sys.stdin = io.StringIO(f"{{{{{var_name}}}}}")
        sys.stdout = io.StringIO()

        fill_command(args)
        result = sys.stdout.getvalue()

        assert '42' in result, f"Variable {var_name} should be set to 42"
        assert 'py:' not in result, "Variable name should not include 'py:' prefix"
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout

# Run the test
test_fill_command_py_prefix_removal()