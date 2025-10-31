import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st

@given(st.just("NBitBase"))
def test_getattr_returns_value(attr_name):
    import importlib
    # Reload the module to ensure clean state
    importlib.reload(npt)
    # Delete NBitBase from module dict to force __getattr__ call
    del npt.__dict__['NBitBase']
    # This should use __getattr__ to retrieve NBitBase
    result = getattr(npt, attr_name)
    assert result is not None

if __name__ == "__main__":
    test_getattr_returns_value()