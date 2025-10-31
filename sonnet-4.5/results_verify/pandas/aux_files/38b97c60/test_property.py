import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, settings
from pandas.core.indexers.objects import FixedWindowIndexer


@given(
    num_values=st.integers(min_value=0, max_value=100),
    window_size=st.integers(min_value=0, max_value=50),
    center=st.booleans(),
    closed=st.sampled_from([None, "left", "right", "both", "neither"]),
    step=st.one_of(st.none(), st.integers(min_value=1, max_value=10))
)
@settings(max_examples=1000)
def test_fixed_window_indexer_invariants(num_values, window_size, center, closed, step):
    indexer = FixedWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(
        num_values=num_values,
        center=center,
        closed=closed,
        step=step
    )

    assert len(start) == len(end)

    for i in range(len(start)):
        assert start[i] <= end[i], f"Invariant violated at index {i}: start[{i}]={start[i]} > end[{i}]={end[i]}, params: num_values={num_values}, window_size={window_size}, center={center}, closed={closed}, step={step}"
        assert 0 <= start[i] <= num_values
        assert 0 <= end[i] <= num_values

if __name__ == "__main__":
    test_fixed_window_indexer_invariants()
    print("Test completed successfully!")