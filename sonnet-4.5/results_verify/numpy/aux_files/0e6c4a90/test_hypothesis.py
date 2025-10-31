from hypothesis import given, strategies as st, settings


@given(st.just(None))
@settings(max_examples=1)
def test_nbitbase_getattr_returns_nbitbase_after_deletion(unused):
    """Property: __getattr__ should return NBitBase when it's been deleted from module dict.

    The numpy.typing module has a custom __getattr__ that specifically checks for
    name == "NBitBase" and is supposed to return it with a deprecation warning.
    This should work even if NBitBase has been deleted from the module dict.
    """
    import numpy.typing as npt
    import importlib
    importlib.reload(npt)

    delattr(npt, 'NBitBase')

    obj = npt.NBitBase
    assert obj is not None

if __name__ == "__main__":
    test_nbitbase_getattr_returns_nbitbase_after_deletion()
    print("Test completed")