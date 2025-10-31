import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env')

import os
import tempfile
import pyximport
from hypothesis import given, strategies as st, settings
import string
from pathlib import Path

@given(st.text(alphabet=string.ascii_letters + string.digits + '_', min_size=1, max_size=50))
@settings(max_examples=100)
def test_get_distutils_extension_bytes_vs_str(modname):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pyx', delete=False) as f:
        f.write("# dummy\n")
        str_path = f.name
        bytes_path = str_path.encode('utf-8')

    try:
        ext_str, args_str = pyximport.get_distutils_extension(modname, str_path)
        ext_bytes, args_bytes = pyximport.get_distutils_extension(modname, bytes_path)
        assert ext_str.name == ext_bytes.name
    finally:
        os.unlink(str_path)

@given(st.text(alphabet=string.ascii_letters + '_', min_size=1, max_size=30))
@settings(max_examples=100)
def test_get_distutils_extension_with_pathlib(modname):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pyx', delete=False) as f:
        f.write("# dummy\n")
        path_str = f.name
        path_obj = Path(path_str)

    try:
        ext_str, args_str = pyximport.get_distutils_extension(modname, path_str)
        ext_path, args_path = pyximport.get_distutils_extension(modname, path_obj)
        assert ext_str.name == ext_path.name
    finally:
        os.unlink(path_str)

if __name__ == "__main__":
    import traceback

    print("Testing bytes path handling:")
    try:
        test_get_distutils_extension_bytes_vs_str()
        print("  PASSED (all 100 examples)")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()

    print("\nTesting Path object handling:")
    try:
        test_get_distutils_extension_with_pathlib()
        print("  PASSED (all 100 examples)")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()