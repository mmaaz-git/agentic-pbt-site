import tempfile
from Cython.Debugger.DebugWriter import CythonDebugWriter, is_valid_tag

print("Case 1: '.0' as regular string")
print(f"is_valid_tag('.0') = {is_valid_tag('.0')}")
with tempfile.TemporaryDirectory() as tmpdir:
    writer = CythonDebugWriter(tmpdir)
    writer.module_name = "test"
    try:
        writer.start('.0')
        print("No crash - tag was accepted")
    except ValueError as e:
        print(f"Crashes with: {e}")

print("\nCase 2: Digit-starting tag '0'")
print(f"is_valid_tag('0') = {is_valid_tag('0')}")
with tempfile.TemporaryDirectory() as tmpdir:
    writer = CythonDebugWriter(tmpdir)
    writer.module_name = "test"
    try:
        writer.start('0')
        print("No crash - tag was accepted")
    except ValueError as e:
        print(f"Crashes with: {e}")

print("\nCase 3: Control character '\\x1f'")
print(f"is_valid_tag('\\x1f') = {is_valid_tag(chr(0x1f))}")
with tempfile.TemporaryDirectory() as tmpdir:
    writer = CythonDebugWriter(tmpdir)
    writer.module_name = "test"
    try:
        writer.start('\x1f')
        print("No crash - tag was accepted")
    except ValueError as e:
        print(f"Crashes with: {e}")