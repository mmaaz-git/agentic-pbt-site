import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.core.indexers import length_of_indexer


@settings(max_examples=1000)
@given(st.data())
def test_length_of_indexer_slice_oracle(data):
    target_len = data.draw(st.integers(min_value=0, max_value=1000))
    target = np.arange(target_len)

    start = data.draw(st.one_of(
        st.none(),
        st.integers(min_value=-target_len*2, max_value=target_len*2)
    ))
    stop = data.draw(st.one_of(
        st.none(),
        st.integers(min_value=-target_len*2, max_value=target_len*2)
    ))
    step = data.draw(st.one_of(
        st.none(),
        st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0)
    ))

    indexer = slice(start, stop, step)

    computed_length = length_of_indexer(indexer, target)
    actual_length = len(target[indexer])

    assert computed_length == actual_length, f"Failed for indexer={indexer}, target_len={target_len}: computed={computed_length}, actual={actual_length}"

if __name__ == "__main__":
    test_length_of_indexer_slice_oracle()
    print("Test completed")