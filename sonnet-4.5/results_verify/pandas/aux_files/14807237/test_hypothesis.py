import numpy as np
from hypothesis import given, strategies as st, settings
from pandas.api.extensions import register_extension_dtype, ExtensionDtype
from pandas.core.dtypes.base import _registry


@given(st.integers(min_value=1, max_value=100))
@settings(max_examples=50)
def test_registry_grows_unbounded_with_duplicate_registrations(n):
    dtype_name = f"test_dtype_{n}"
    initial_count = len(_registry.dtypes)

    class MyDtype(ExtensionDtype):
        name = dtype_name
        type = int

        @classmethod
        def construct_array_type(cls):
            from pandas.core.arrays import ExtensionArray
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

    for i in range(n):
        register_extension_dtype(MyDtype)

    final_count = len(_registry.dtypes)
    growth = final_count - initial_count

    assert growth == n
    print(f"Test with n={n}: Registry grew by {growth} (expected {n})")

if __name__ == "__main__":
    # Run hypothesis test
    test_registry_grows_unbounded_with_duplicate_registrations()