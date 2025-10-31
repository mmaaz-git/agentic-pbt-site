import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st

@given(st.just("NBitBase"))
def test_getattr_returns_value(attr_name):
    import importlib
    importlib.reload(npt)
    del npt.__dict__['NBitBase']
    result = getattr(npt, attr_name)
    assert result is not None

# Run the test
test_getattr_returns_value("NBitBase")
print("Test completed")