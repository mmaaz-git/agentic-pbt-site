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

    assert final_size - initial_size == 1, f"Expected only 1 new entry, but got {final_size - initial_size} after {n_registrations} registrations"


if __name__ == "__main__":
    # Run the property-based test
    print("Running Hypothesis property-based test...")
    test_registration_idempotency()
    print("All tests passed!")

    # Also run manual tests with specific values
    print("\nManual tests with specific values:")
    for n in [1, 2, 3, 5]:
        print(f"Testing with n_registrations={n}")

        class TestDtype(ExtensionDtype):
            name = f"test_dtype_manual_{n}"

            @property
            def type(self):
                return object

            @classmethod
            def construct_array_type(cls):
                from pandas.core.arrays import ExtensionArray
                return ExtensionArray

        initial_size = len(_registry.dtypes)

        for _ in range(n):
            register_extension_dtype(TestDtype)

        final_size = len(_registry.dtypes)

        if final_size - initial_size == 1:
            print(f"  PASSED")
        else:
            print(f"  FAILED: Expected only 1 new entry, but got {final_size - initial_size}")