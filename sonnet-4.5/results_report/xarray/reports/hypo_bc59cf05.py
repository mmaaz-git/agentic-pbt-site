from hypothesis import given, strategies as st, settings
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import OPTIONS


@given(st.sampled_from(["all", "minimal", "exact"]))
@settings(max_examples=100)
def test_hash_stability_across_options_change(val):
    obj = CombineKwargDefault(name="test", old="old_value", new="new_value")

    original_option = OPTIONS["use_new_combine_kwarg_defaults"]

    try:
        OPTIONS["use_new_combine_kwarg_defaults"] = False
        hash1 = hash(obj)

        OPTIONS["use_new_combine_kwarg_defaults"] = True
        hash2 = hash(obj)

        assert hash1 == hash2
    finally:
        OPTIONS["use_new_combine_kwarg_defaults"] = original_option


if __name__ == "__main__":
    test_hash_stability_across_options_change()