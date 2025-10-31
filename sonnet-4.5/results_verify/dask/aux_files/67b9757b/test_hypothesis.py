#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from dask.diagnostics.profile_visualize import unquote

@given(
    expr=st.recursive(
        st.one_of(st.integers(), st.text(), st.floats(allow_nan=False)),
        lambda children: st.one_of(
            st.tuples(st.just(tuple), st.lists(children, max_size=3)),
            st.tuples(st.just(list), st.lists(children, max_size=3)),
            st.tuples(st.just(set), st.lists(children, max_size=3)),
        ),
        max_leaves=10
    )
)
@settings(max_examples=100)
def test_unquote_idempotence(expr):
    try:
        result1 = unquote(expr)
        result2 = unquote(result1)
        assert result1 == result2
    except Exception as e:
        print(f"Failed on expr: {expr}")
        print(f"Exception: {e}")
        raise

print("Running hypothesis test...")
test_unquote_idempotence()
print("Test completed")
