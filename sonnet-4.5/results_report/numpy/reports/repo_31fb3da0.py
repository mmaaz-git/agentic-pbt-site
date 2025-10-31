import tempfile
from unittest.mock import patch

import numpy as np

# Mock sysconfig.get_config_var to return None (simulating missing EXT_SUFFIX)
with tempfile.TemporaryDirectory() as tmpdir:
    with patch('sysconfig.get_config_var', return_value=None):
        try:
            # This should crash with a TypeError
            np.ctypeslib.load_library('mylib', tmpdir)
        except Exception as e:
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")