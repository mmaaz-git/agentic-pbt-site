import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, assume, strategies as st
from Cython.Tempita import Template

@given(st.text(alphabet='0123456789+-*/', min_size=1, max_size=10))
def test_syntaxerror_includes_position(expr):
    assume('{{' not in expr and '}}' not in expr)

    content = f"Line 1\nLine 2\n{{{{{expr}}}}}"
    template = Template(content)

    try:
        template.substitute({})
    except SyntaxError as e:
        error_msg = str(e)
        assert 'line' in error_msg.lower() and 'column' in error_msg.lower(), \
            f"SyntaxError message lacks position info: '{error_msg}'"

if __name__ == "__main__":
    # Run the test
    test_syntaxerror_includes_position()