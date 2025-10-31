from hypothesis import given, strategies as st, settings
from Cython.Debugger.DebugWriter import CythonDebugWriter
import tempfile
import shutil
import os


@given(st.text(min_size=1))
@settings(max_examples=500)
def test_cython_debug_writer_start_end_pairing(tag_name):
    tmpdir = tempfile.mkdtemp()
    try:
        writer = CythonDebugWriter(tmpdir)
        writer.module_name = 'test_module'

        writer.start('Module')
        writer.start(tag_name)
        writer.end(tag_name)
        writer.serialize()
    finally:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

# Run the test
test_cython_debug_writer_start_end_pairing()