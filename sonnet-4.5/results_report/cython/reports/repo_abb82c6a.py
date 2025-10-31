from Cython.Debugger.DebugWriter import CythonDebugWriter, is_valid_tag
import tempfile
import shutil

# Test case 1: Control character '\x08'
print("Test 1: Control character '\\x08'")
print(f"is_valid_tag('\\x08') returns: {is_valid_tag('\x08')}")

tmpdir = tempfile.mkdtemp()
try:
    writer = CythonDebugWriter(tmpdir)
    writer.module_name = 'test_module'
    writer.start('Module')
    writer.start('\x08')  # This should crash
    writer.end('\x08')
    writer.serialize()
    print("No error - unexpected!")
except Exception as e:
    print(f"Error: {e}")
finally:
    shutil.rmtree(tmpdir)

print()

# Test case 2: Tag name starting with digit '0'
print("Test 2: Tag name starting with digit '0'")
print(f"is_valid_tag('0') returns: {is_valid_tag('0')}")

tmpdir = tempfile.mkdtemp()
try:
    writer = CythonDebugWriter(tmpdir)
    writer.module_name = 'test_module'
    writer.start('Module')
    writer.start('0')  # This should crash
    writer.end('0')
    writer.serialize()
    print("No error - unexpected!")
except Exception as e:
    print(f"Error: {e}")
finally:
    shutil.rmtree(tmpdir)