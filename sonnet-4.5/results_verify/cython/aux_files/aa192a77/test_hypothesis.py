import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from distutils.dist import Distribution
from Cython.Distutils import build_ext

@settings(max_examples=100)
@given(st.text(min_size=1, max_size=100))
def test_directives_type_validation(directive_value):
    """
    Property: cython_directives should be validated or converted to dict
    """
    dist = Distribution()
    cmd = build_ext(dist)
    cmd.initialize_options()

    cmd.cython_directives = directive_value
    cmd.finalize_options()

    result = cmd.cython_directives
    assert isinstance(result, dict), \
        f"cython_directives should be dict after finalize, got {type(result)}"

# Run the test
test_directives_type_validation()