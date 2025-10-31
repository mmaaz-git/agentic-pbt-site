import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from attr._cmp import cmp_using

@given(st.sampled_from([lambda a, b: a < b, lambda a, b: a > b]))
def test_cmp_using_error_message_grammar(lt_func):
    try:
        cmp_using(lt=lt_func)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        assert "must be defined in order" in error_msg, \
            f"Error message has typos: {error_msg}"

if __name__ == "__main__":
    test_cmp_using_error_message_grammar()