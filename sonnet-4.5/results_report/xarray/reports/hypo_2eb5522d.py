import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from xarray.core.formatting_html import _load_static_files


@given(st.text())
def test_cache_immutability(mutation_data):
    result1 = _load_static_files()
    original_length = len(result1)

    result1.append(mutation_data)

    result2 = _load_static_files()

    assert len(result2) == original_length, \
        "Cached result should not be affected by mutations to previous return values"

if __name__ == "__main__":
    test_cache_immutability()