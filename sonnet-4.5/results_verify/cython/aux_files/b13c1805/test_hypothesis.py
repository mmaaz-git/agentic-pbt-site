import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from Cython.Tempita import Template, TemplateError


@given(st.text(alphabet=' \t', min_size=0, max_size=5))
@settings(max_examples=100)
def test_parse_def_whitespace_only(whitespace):
    content = f"{{{{def{whitespace}}}}}{{{{enddef}}}}"

    print(f"Testing with whitespace: {repr(whitespace)}, content: {repr(content)}")

    try:
        template = Template(content)
        assert False, "Should raise TemplateError"
    except Exception as e:
        print(f"Got exception type: {type(e).__name__}, message: {e}")
        assert isinstance(e, TemplateError), f"Should be TemplateError, got {type(e).__name__}"

# Run the test
if __name__ == "__main__":
    test_parse_def_whitespace_only()