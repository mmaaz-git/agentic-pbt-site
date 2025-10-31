from hypothesis import given, strategies as st, settings
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options


@given(
    name=st.text(min_size=1),
    old=st.text(),
    new=st.one_of(st.none(), st.text())
)
@settings(max_examples=1000)
def test_combine_kwarg_default_hash_immutable(name, old, new):
    obj = CombineKwargDefault(name=name, old=old, new=new)

    hash1 = hash(obj)

    with set_options(use_new_combine_kwarg_defaults=True):
        hash2 = hash(obj)

    with set_options(use_new_combine_kwarg_defaults=False):
        hash3 = hash(obj)

    assert hash1 == hash2 == hash3, f"Hash changed! hash1={hash1}, hash2={hash2}, hash3={hash3}"


if __name__ == "__main__":
    test_combine_kwarg_default_hash_immutable()