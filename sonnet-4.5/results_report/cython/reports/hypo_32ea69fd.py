import os
import tempfile
import sys
from hypothesis import given, strategies as st

# Add the pyximport to path if needed
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')
import pyximport

@given(st.just(None))
def test_handle_special_build_with_only_setup_args(x):
    with tempfile.TemporaryDirectory() as tmpdir:
        pyx_file = os.path.join(tmpdir, 'test.pyx')
        pyxbld_file = os.path.join(tmpdir, 'test.pyxbld')

        with open(pyx_file, 'w') as f:
            f.write('def hello(): return "world"')

        with open(pyxbld_file, 'w') as f:
            f.write('def make_setup_args():\n    return {"script_args": ["--verbose"]}')

        ext, setup_args = pyximport.handle_special_build('test', pyx_file)
        assert isinstance(setup_args, dict)

if __name__ == "__main__":
    test_handle_special_build_with_only_setup_args()