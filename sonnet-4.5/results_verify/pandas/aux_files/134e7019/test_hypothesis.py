import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, settings
from pandas.core.indexers import length_of_indexer

@given(
    target_list=st.lists(st.integers(), min_size=0, max_size=50),
    slice_start=st.integers(min_value=-60, max_value=60) | st.none(),
    slice_stop=st.integers(min_value=-60, max_value=60) | st.none(),
    slice_step=st.integers(min_value=-10, max_value=10).filter(lambda x: x != 0) | st.none(),
)
@settings(max_examples=500)
def test_length_of_indexer_slice(target_list, slice_start, slice_stop, slice_step):
    indexer = slice(slice_start, slice_stop, slice_step)
    expected_length = len(target_list[indexer])
    computed_length = length_of_indexer(indexer, target_list)
    assert computed_length == expected_length, f"Expected {expected_length}, got {computed_length} for target_list={target_list}, slice({slice_start}, {slice_stop}, {slice_step})"

if __name__ == "__main__":
    test_length_of_indexer_slice()
    print("Test passed!")