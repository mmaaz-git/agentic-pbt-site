from hypothesis import given, strategies as st, assume
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options

@given(
    name=st.text(min_size=1),
    old=st.text(),
    new=st.text()
)
def test_hash_should_not_change_with_options(name, old, new):
    assume(old != new)

    obj = CombineKwargDefault(name=name, old=old, new=new)

    with set_options(use_new_combine_kwarg_defaults=False):
        hash1 = hash(obj)

    with set_options(use_new_combine_kwarg_defaults=True):
        hash2 = hash(obj)

    assert hash1 == hash2, f"Hash changed from {hash1} to {hash2} for name='{name}', old='{old}', new='{new}'"

if __name__ == "__main__":
    # Run the property test
    test_hash_should_not_change_with_options()