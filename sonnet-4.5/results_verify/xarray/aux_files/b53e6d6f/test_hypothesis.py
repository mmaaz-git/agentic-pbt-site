import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from xarray.core.formatting_html import _load_static_files


@given(st.integers(min_value=0, max_value=1))
def test_load_static_files_cache_mutation(index):
    result1 = _load_static_files()
    original_first_item = result1[0]

    result1[index] = "MUTATED"

    result2 = _load_static_files()

    assert result2[0] == original_first_item, f"Cache returned mutated result. Expected original content, got: {result2[0][:50] if len(result2[0]) > 50 else result2[0]}"

if __name__ == "__main__":
    test_load_static_files_cache_mutation()
    print("Test completed")