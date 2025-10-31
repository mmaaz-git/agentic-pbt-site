from unittest.mock import patch
from hypothesis import given, settings, strategies as st
import numpy as np
import tempfile

@given(
    libname=st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=10)
)
@settings(max_examples=100)
def test_load_library_handles_none_ext_suffix(libname):
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('sysconfig.get_config_var', return_value=None):
            try:
                np.ctypeslib.load_library(libname, tmpdir)
            except OSError as e:
                if "no file with expected extension" in str(e):
                    pass
                else:
                    raise
            except TypeError as e:
                if "unsupported operand type" in str(e) or "can only concatenate" in str(e):
                    assert False, f"Bug: load_library crashes when EXT_SUFFIX is None: {e}"
                else:
                    raise

# Run the test
test_load_library_handles_none_ext_suffix()