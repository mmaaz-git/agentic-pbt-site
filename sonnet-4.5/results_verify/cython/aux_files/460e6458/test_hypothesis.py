import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from Cython.Tempita._looper import looper


@given(st.lists(st.integers(), min_size=1, max_size=10))
@settings(max_examples=100)
def test_looper_last_group_with_callable_getter(values):
    class Item:
        def __init__(self, val):
            self.val = val

    items = [Item(v) for v in values]

    for loop, item in looper(items):
        if loop.last:
            try:
                result = loop.last_group('.val')
                assert result == True
            except AttributeError as e:
                assert False, f"last_group should not crash on last item: {e}"

if __name__ == "__main__":
    test_looper_last_group_with_callable_getter()
    print("Hypothesis test completed")