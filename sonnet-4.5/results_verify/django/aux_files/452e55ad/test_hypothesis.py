import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django.template
from hypothesis import given, strategies as st


@given(st.integers(min_value=0, max_value=1000))
def test_variable_trailing_dot_inconsistency(n):
    var_string = f'{n}.'
    v = django.template.Variable(var_string)

    if v.literal is not None and v.lookups is not None:
        assert False, f'Variable should not have both literal and lookups set'

# Run a simple test
if __name__ == "__main__":
    test_variable_trailing_dot_inconsistency()