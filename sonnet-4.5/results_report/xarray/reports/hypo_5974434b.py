from hypothesis import given, strategies as st, settings
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS


@given(
    name=st.text(min_size=1),
    old=st.text(),
    new=st.text(),
)
@settings(max_examples=100)
def test_hash_remains_constant_during_object_lifetime(name, old, new):
    """
    Property: An object's hash must remain constant during its lifetime.
    """
    obj = CombineKwargDefault(name=name, old=old, new=new)

    original_hash = hash(obj)
    original_option = OPTIONS["use_new_combine_kwarg_defaults"]

    OPTIONS["use_new_combine_kwarg_defaults"] = not original_option
    new_hash = hash(obj)
    OPTIONS["use_new_combine_kwarg_defaults"] = original_option

    assert original_hash == new_hash, (
        f"Hash changed when global OPTIONS changed! "
        f"Before: {original_hash}, After: {new_hash}. "
        f"This violates Python's hash invariant."
    )


if __name__ == "__main__":
    test_hash_remains_constant_during_object_lifetime()