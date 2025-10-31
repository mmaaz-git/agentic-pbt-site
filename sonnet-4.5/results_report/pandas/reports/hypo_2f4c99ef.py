import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, settings
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

@given(st.data())
@settings(max_examples=500)
def test_arrow_extension_array_fillna_length(data):
    values = data.draw(st.lists(st.one_of(st.integers(min_value=-1000, max_value=1000), st.none()), min_size=1, max_size=100))
    arr = ArrowExtensionArray(pa.array(values))
    fill_value = data.draw(st.integers(min_value=-1000, max_value=1000))

    result = arr.fillna(fill_value)

    assert len(result) == len(arr)

if __name__ == "__main__":
    # Run the test
    test_arrow_extension_array_fillna_length()