import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=('Cc', 'Cs')), min_size=1)))
def test_parse_list_bracket_delimited(items):
    s = '[' + ', '.join(items) + ']'
    result = parse_list(s)
    assert isinstance(result, list)

# Run the test
test_parse_list_bracket_delimited()