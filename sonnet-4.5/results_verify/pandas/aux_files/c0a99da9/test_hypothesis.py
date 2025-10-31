import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS


@given(st.data())
def test_argsort_defaults_kind_should_not_be_duplicated(data):
    assert "kind" in ARGSORT_DEFAULTS
    kind_value = ARGSORT_DEFAULTS["kind"]

    assert kind_value is not None, (
        f"ARGSORT_DEFAULTS['kind'] should have a default value, "
        f"but got None. This appears to be due to duplicate assignment "
        f"where 'kind' is first set to 'quicksort' then overwritten to None."
    )

# Run the test
test_argsort_defaults_kind_should_not_be_duplicated()