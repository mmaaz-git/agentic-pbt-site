from pandas.api.extensions import ExtensionDtype, register_extension_dtype
from pandas.core.dtypes.base import _registry
from hypothesis import given, strategies as st


@given(st.integers(min_value=1, max_value=10))
def test_registration_idempotency(n_registrations):
    class TestDtype(ExtensionDtype):
        name = "test_dtype_idempotent"

        @property
        def type(self):
            return object

        @classmethod
        def construct_array_type(cls):
            from pandas.core.arrays import ExtensionArray
            return ExtensionArray

    initial_size = len(_registry.dtypes)

    for _ in range(n_registrations):
        register_extension_dtype(TestDtype)

    final_size = len(_registry.dtypes)

    assert final_size - initial_size == 1, f"Expected 1 new registration, got {final_size - initial_size}"


if __name__ == "__main__":
    test_registration_idempotency()