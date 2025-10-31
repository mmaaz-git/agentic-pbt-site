import sys
import string
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Tempita import Template

@given(st.dictionaries(
    keys=st.text(alphabet=string.ascii_letters + '_', min_size=1).filter(str.isidentifier),
    values=st.integers(),
    min_size=1, max_size=3
))
def test_substitute_does_not_mutate_input(user_vars):
    if not user_vars:
        user_vars = {'x': 1}

    var_to_use = list(user_vars.keys())[0]
    template = Template(f'{{{{{var_to_use}}}}}', namespace={'other': 999})
    original_vars = user_vars.copy()

    template.substitute(user_vars)

    assert user_vars == original_vars, f"Dictionary was mutated! Original: {original_vars}, After: {user_vars}"

if __name__ == "__main__":
    test_substitute_does_not_mutate_input()
