import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Tempita import Template, TemplateError
import pytest

@given(st.text(alphabet=' \t', min_size=0, max_size=5))
def test_def_with_no_signature_raises_template_error(whitespace):
    content = f"{{{{def{whitespace}}}}}{{{{enddef}}}}"

    print(f"Testing: {repr(content)}")

    with pytest.raises(TemplateError):
        template = Template(content)

# Run the test
if __name__ == "__main__":
    # Import hypothesis testing
    test_def_with_no_signature_raises_template_error()