import sys
import os
import tempfile
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from pyximport.pyximport import handle_special_build
from hypothesis import given, strategies as st, settings


@settings(max_examples=100)
@given(
    has_make_ext=st.booleans(),
    has_make_setup_args=st.booleans()
)
def test_handle_special_build_combinations(has_make_ext, has_make_setup_args):
    with tempfile.TemporaryDirectory() as tmpdir:
        pyxfile = os.path.join(tmpdir, "test.pyx")
        pyxbld = os.path.join(tmpdir, "test.pyxbld")

        open(pyxfile, 'w').close()

        pyxbld_content = ""
        if has_make_ext:
            pyxbld_content += """
from distutils.extension import Extension
def make_ext(modname, pyxfilename):
    return Extension(name=modname, sources=[pyxfilename])
"""
        if has_make_setup_args:
            pyxbld_content += """
def make_setup_args():
    return {'script_args': ['--verbose']}
"""

        if pyxbld_content:
            with open(pyxbld, 'w') as f:
                f.write(pyxbld_content)

            try:
                ext, setup_args = handle_special_build("test", pyxfile)
                print(f"Success for make_ext={has_make_ext}, make_setup_args={has_make_setup_args}")
            except AttributeError as e:
                print(f"AttributeError for make_ext={has_make_ext}, make_setup_args={has_make_setup_args}: {e}")
                return
            except Exception as e:
                print(f"Other error for make_ext={has_make_ext}, make_setup_args={has_make_setup_args}: {e}")
                return
        else:
            # No pyxbld file created - this is expected behavior
            ext, setup_args = handle_special_build("test", pyxfile)
            assert ext is None and setup_args == {}
            print(f"No pyxbld, as expected ext={ext}, setup_args={setup_args}")

# Run the test
test_handle_special_build_combinations()