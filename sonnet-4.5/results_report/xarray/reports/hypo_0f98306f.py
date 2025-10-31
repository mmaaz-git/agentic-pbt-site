import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS


@given(
    name=st.text(min_size=1),
    old=st.text(),
    new=st.one_of(st.none(), st.text())
)
@settings(max_examples=1000)
def test_combine_kwarg_hash_stable_across_options_change(name, old, new):
    obj = CombineKwargDefault(name=name, old=old, new=new)

    original_setting = OPTIONS["use_new_combine_kwarg_defaults"]
    hash1 = hash(obj)

    OPTIONS["use_new_combine_kwarg_defaults"] = not original_setting
    hash2 = hash(obj)

    OPTIONS["use_new_combine_kwarg_defaults"] = original_setting

    assert hash1 == hash2, f"Hash changed from {hash1} to {hash2} when OPTIONS changed"


# Run the test
if __name__ == "__main__":
    test_combine_kwarg_hash_stable_across_options_change()