import os
import tempfile
from hypothesis import given, strategies as st, settings
import pyximport


@given(st.booleans())
@settings(max_examples=50)
def test_handle_special_build_with_setup_args_only(include_make_ext):
    with tempfile.TemporaryDirectory() as tmpdir:
        pyxfile = os.path.join(tmpdir, 'test.pyx')
        pyxbld_file = os.path.join(tmpdir, 'test.pyxbld')

        with open(pyxfile, 'w') as f:
            f.write('# cython code')

        if include_make_ext:
            pyxbld_content = '''
from distutils.extension import Extension

def make_ext(modname, pyxfilename):
    return Extension(name=modname, sources=[pyxfilename])

def make_setup_args():
    return {'extra_compile_args': ['-O3']}
'''
        else:
            pyxbld_content = '''
def make_setup_args():
    return {'extra_compile_args': ['-O3']}
'''

        with open(pyxbld_file, 'w') as f:
            f.write(pyxbld_content)

        ext, setup_args = pyximport.handle_special_build('test', pyxfile)


if __name__ == '__main__':
    test_handle_special_build_with_setup_args_only()