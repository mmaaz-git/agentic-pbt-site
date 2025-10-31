from Cython.Debugger.DebugWriter import CythonDebugWriter, is_valid_tag
import tempfile
import shutil

tmpdir = tempfile.mkdtemp()

writer = CythonDebugWriter(tmpdir)
writer.module_name = 'test_module'
writer.start('Module')

print(f"is_valid_tag('\\x08') returns: {is_valid_tag('\x08')}")
try:
    writer.start('\x08')
    print("No error - bug confirmed!")
except ValueError as e:
    print(f"Error raised: {e}")

shutil.rmtree(tmpdir)