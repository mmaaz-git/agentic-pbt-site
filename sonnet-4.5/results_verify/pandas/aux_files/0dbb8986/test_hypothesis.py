import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pytest
from hypothesis import given, strategies as st, settings
from pandas.compat.numpy import function as nv


@given(st.sampled_from(['quicksort', 'mergesort', 'heapsort', 'stable']))
@settings(max_examples=4)
def test_validate_argsort_with_kind_parameter(kind):
    validator = nv.validate_argsort
    try:
        validator((), {"kind": kind})
        print(f"✓ kind='{kind}' passed validation")
    except ValueError as e:
        print(f"✗ kind='{kind}' raised ValueError: {e}")

if __name__ == "__main__":
    test_validate_argsort_with_kind_parameter()