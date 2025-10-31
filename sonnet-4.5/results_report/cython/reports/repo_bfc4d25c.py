import tempfile
import shutil
from Cython.Debugger.DebugWriter import CythonDebugWriter

tmpdir = tempfile.mkdtemp()

try:
    writer = CythonDebugWriter(tmpdir)
    writer.tb.start('Module', {})

    # Try to add an entry with a control character in an attribute value
    writer.add_entry('a', a='\x1f')

    writer.tb.end('Module')
    writer.tb.end('cython_debug')
    writer.tb.close()
    print("No crash - unexpected")
except ValueError as e:
    print(f"Crashed with ValueError: {e}")
except Exception as e:
    print(f"Crashed with {type(e).__name__}: {e}")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)