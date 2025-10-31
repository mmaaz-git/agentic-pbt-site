from hypothesis import given, strategies as st
from dask.diagnostics.profile_visualize import unquote

@given(expr=st.recursive(
    st.one_of(st.integers(), st.text(), st.none()),
    lambda children: st.tuples(
        st.sampled_from([tuple, list, set, dict]),
        st.lists(children, max_size=3)
    ),
    max_leaves=10
))
def test_unquote_idempotence(expr):
    once = unquote(expr)
    twice = unquote(once)
    assert once == twice

if __name__ == "__main__":
    test_unquote_idempotence()