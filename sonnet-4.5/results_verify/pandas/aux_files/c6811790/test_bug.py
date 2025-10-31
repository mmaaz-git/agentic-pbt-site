from pandas.core.dtypes.base import StorageExtensionDtype
from hypothesis import given, strategies as st, settings


class TestStorageDtype(StorageExtensionDtype):
    name = "test_storage"

    @classmethod
    def construct_array_type(cls):
        return None


@given(storage=st.one_of(st.none(), st.text(min_size=1, max_size=20)))
@settings(max_examples=500)
def test_storage_dtype_hash_consistency_with_string(storage):
    dtype = TestStorageDtype(storage=storage)

    if dtype == dtype.name:
        assert hash(dtype) == hash(dtype.name), f"Hash mismatch for storage={storage!r}: hash(dtype)={hash(dtype)}, hash(name)={hash(dtype.name)}"


@given(
    storage1=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
    storage2=st.one_of(st.none(), st.text(min_size=1, max_size=20))
)
@settings(max_examples=500)
def test_storage_dtype_equality_transitivity(storage1, storage2):
    dtype1 = TestStorageDtype(storage=storage1)
    dtype2 = TestStorageDtype(storage=storage2)
    name_str = dtype1.name

    if dtype1 == name_str and name_str == dtype2:
        assert dtype1 == dtype2, f"Transitivity violation: storage1={storage1!r}, storage2={storage2!r}"


if __name__ == "__main__":
    print("Testing hash consistency...")
    try:
        test_storage_dtype_hash_consistency_with_string()
        print("Hash consistency test passed!")
    except AssertionError as e:
        print(f"Hash consistency test FAILED: {e}")

    print("\nTesting equality transitivity...")
    try:
        test_storage_dtype_equality_transitivity()
        print("Transitivity test passed!")
    except AssertionError as e:
        print(f"Transitivity test FAILED: {e}")