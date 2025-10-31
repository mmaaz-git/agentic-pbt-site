import pandas as pd
from pandas.api.extensions import register_extension_dtype, ExtensionDtype

@register_extension_dtype
class FirstDtype(ExtensionDtype):
    name = 'duplicatename'
    type = object
    _metadata = ()

    @classmethod
    def construct_array_type(cls):
        from pandas.core.arrays import PandasArray
        return PandasArray

@register_extension_dtype
class SecondDtype(ExtensionDtype):
    name = 'duplicatename'
    type = object
    _metadata = ()

    @classmethod
    def construct_array_type(cls):
        from pandas.core.arrays import PandasArray
        return PandasArray

retrieved_dtype = pd.api.types.pandas_dtype('duplicatename')
print(f"Retrieved dtype type: {type(retrieved_dtype)}")
print(f"Is FirstDtype: {isinstance(retrieved_dtype, FirstDtype)}")
print(f"Is SecondDtype: {isinstance(retrieved_dtype, SecondDtype)}")

# The bug report expects this behavior
assert isinstance(retrieved_dtype, FirstDtype)
assert not isinstance(retrieved_dtype, SecondDtype)
print("Bug confirmed: Second registration is silently ignored!")