from hypothesis import given, strategies as st, assume
import pandas as pd
from pandas.api.extensions import register_extension_dtype, ExtensionDtype


@given(dtype_name=st.text(alphabet=st.characters(whitelist_categories=('L',), min_codepoint=97, max_codepoint=122), min_size=1, max_size=30))
def test_register_extension_dtype_idempotence(dtype_name):
    assume(dtype_name.isidentifier())

    @register_extension_dtype
    class TestDtype1(ExtensionDtype):
        name = dtype_name
        type = object
        _metadata = ()

        @classmethod
        def construct_array_type(cls):
            from pandas.core.arrays import PandasArray
            return PandasArray

    @register_extension_dtype
    class TestDtype2(ExtensionDtype):
        name = dtype_name
        type = object
        _metadata = ()

        @classmethod
        def construct_array_type(cls):
            from pandas.core.arrays import PandasArray
            return PandasArray

    dtype = pd.api.types.pandas_dtype(dtype_name)
    assert isinstance(dtype, TestDtype2), f"Expected TestDtype2 but got {type(dtype)}"

# Run the test with a simple example
# We need to call the wrapped function directly
import sys

dtype_name = 'a'

@register_extension_dtype
class TestDtype1(ExtensionDtype):
    name = dtype_name
    type = object
    _metadata = ()

    @classmethod
    def construct_array_type(cls):
        from pandas.core.arrays import PandasArray
        return PandasArray

@register_extension_dtype
class TestDtype2(ExtensionDtype):
    name = dtype_name
    type = object
    _metadata = ()

    @classmethod
    def construct_array_type(cls):
        from pandas.core.arrays import PandasArray
        return PandasArray

dtype = pd.api.types.pandas_dtype(dtype_name)
try:
    assert isinstance(dtype, TestDtype2), f"Expected TestDtype2 but got {type(dtype)}"
    print("Test passed - bug not present")
except AssertionError as e:
    print(f"Test failed as expected - bug confirmed: {e}")