from pandas.api.extensions import ExtensionDtype, register_extension_dtype
from pandas.core.dtypes.base import _registry


class MyDtype(ExtensionDtype):
    name = "mydtype"

    @property
    def type(self):
        return object

    @classmethod
    def construct_array_type(cls):
        from pandas.core.arrays import ExtensionArray
        return ExtensionArray


initial_size = len(_registry.dtypes)
print(f"Initial registry size: {initial_size}")

register_extension_dtype(MyDtype)
print(f"After first registration: {len(_registry.dtypes)}")

register_extension_dtype(MyDtype)
print(f"After second registration: {len(_registry.dtypes)}")

print(f"\nDuplicate entries: {len(_registry.dtypes) - initial_size - 1}")