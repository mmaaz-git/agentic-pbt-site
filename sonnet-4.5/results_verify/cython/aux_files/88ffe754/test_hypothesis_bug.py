import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, assume, strategies as st
import string
from Cython.Tempita import Template

@given(st.dictionaries(
    keys=st.text(alphabet=string.ascii_letters + '_', min_size=1).filter(str.isidentifier),
    values=st.integers(),
    min_size=0, max_size=5
))
def test_substitute_doesnt_mutate_input(vars_dict):
    assume('__template_name__' not in vars_dict)

    template = Template("{{x}}", name="test")
    original_keys = set(vars_dict.keys())

    result = template.substitute(vars_dict)

    assert set(vars_dict.keys()) == original_keys, \
        f"substitute() mutated input dict: added {set(vars_dict.keys()) - original_keys}"

# Run the test
if __name__ == "__main__":
    test_substitute_doesnt_mutate_input()