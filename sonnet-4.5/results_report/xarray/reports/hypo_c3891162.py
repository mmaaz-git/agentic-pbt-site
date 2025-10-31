import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
from xarray.coding.strings import encode_string_array, decode_bytes_array

@given(
    strings=arrays(
        dtype=object,
        shape=st.integers(min_value=1, max_value=20),
        elements=st.text(min_size=0, max_size=50)
    )
)
def test_string_encode_decode_roundtrip(strings):
    encoded = encode_string_array(strings, encoding='utf-8')
    decoded = decode_bytes_array(encoded, encoding='utf-8')
    assert np.array_equal(decoded, strings)

if __name__ == "__main__":
    test_string_encode_decode_roundtrip()