import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import string
from Cython.Tempita import Template

@given(st.text(alphabet=string.ascii_letters + '_', min_size=1).filter(str.isidentifier),
       st.integers(min_value=0, max_value=100),
       st.integers(min_value=101, max_value=200))
def test_substitute_args_override_namespace(var_name, namespace_value, substitute_value):
    content = f"{{{{{var_name}}}}}"
    template = Template(content, namespace={var_name: namespace_value})
    result = template.substitute({var_name: substitute_value})

    assert result == str(substitute_value), f"Expected {substitute_value}, got {result}"

# Run the test
test_substitute_args_override_namespace()