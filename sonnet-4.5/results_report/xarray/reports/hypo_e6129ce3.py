from hypothesis import given, settings, strategies as st
from xarray.util.deprecation_helpers import CombineKwargDefault
from xarray.core.options import set_options


@given(st.text(min_size=1, max_size=50))
@settings(max_examples=200)
def test_dask_tokenize_immutability(name):
    obj = CombineKwargDefault(name=name, old="old_value", new="new_value")

    with set_options(use_new_combine_kwarg_defaults=False):
        token1 = obj.__dask_tokenize__()

    with set_options(use_new_combine_kwarg_defaults=True):
        token2 = obj.__dask_tokenize__()

    assert token1 == token2, f"Dask token changed when OPTIONS changed: {token1} != {token2}"


if __name__ == "__main__":
    test_dask_tokenize_immutability()