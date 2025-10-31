import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from dask.diagnostics.profile_visualize import unquote
from dask.core import istask


@given(st.lists(st.lists(st.tuples(st.text(), st.integers())), min_size=0))
def test_unquote_dict_no_crash(items):
    task = (dict, items)
    if istask(task):
        try:
            result = unquote(task)
        except (ValueError, IndexError) as e:
            raise AssertionError(f"unquote crashed with {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_unquote_dict_no_crash()