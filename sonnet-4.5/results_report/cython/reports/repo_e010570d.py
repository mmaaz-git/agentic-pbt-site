import os
import tempfile
import pyximport

with tempfile.TemporaryDirectory() as tmpdir:
    pyxfile = os.path.join(tmpdir, 'test.pyx')
    pyxbld_file = os.path.join(tmpdir, 'test.pyxbld')

    with open(pyxfile, 'w') as f:
        f.write('# cython code')

    with open(pyxbld_file, 'w') as f:
        f.write('''def make_setup_args():
    return {'extra_compile_args': ['-O3']}
''')

    ext, setup_args = pyximport.handle_special_build('test', pyxfile)