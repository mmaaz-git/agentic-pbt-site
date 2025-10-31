import numpy as np
from pandas.api.extensions import register_extension_dtype, ExtensionDtype
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.base import _registry


class MyDtype(ExtensionDtype):
    name = "mydtype"
    type = int

    @classmethod
    def construct_array_type(cls):
        class MyArray(ExtensionArray):
            dtype = cls()

            def __init__(self, data):
                self._data = np.array(data)

            def __len__(self):
                return len(self._data)

            def __getitem__(self, item):
                return self._data[item]

            def isna(self):
                return np.zeros(len(self._data), dtype=bool)

        return MyArray


initial = len(_registry.dtypes)
print(f"Initial registry size: {initial}")

register_extension_dtype(MyDtype)
after_first = len(_registry.dtypes)
print(f"After first registration: {after_first} (growth: {after_first - initial})")

register_extension_dtype(MyDtype)
after_second = len(_registry.dtypes)
print(f"After second registration: {after_second} (growth: {after_second - initial})")

register_extension_dtype(MyDtype)
final = len(_registry.dtypes)
print(f"After third registration: {final} (growth: {final - initial})")

print(f"\nRegistry grew by {final - initial} (expected 1, got {final - initial})")

# Check if all three entries are the same class
print(f"\nAll registered dtypes with name 'mydtype':")
count = 0
for dtype in _registry.dtypes:
    if hasattr(dtype, 'name') and dtype.name == 'mydtype':
        count += 1
        print(f"  - Found dtype: {dtype}, id: {id(dtype)}")
print(f"Total count: {count}")