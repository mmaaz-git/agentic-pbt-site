import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from xarray.compat.pdcompat import default_precision_timestamp


@given(st.datetimes())
@settings(max_examples=1000)
def test_default_precision_timestamp_unit(dt):
    result = default_precision_timestamp(dt)
    assert result.unit == 'ns'

if __name__ == "__main__":
    test_default_precision_timestamp_unit()