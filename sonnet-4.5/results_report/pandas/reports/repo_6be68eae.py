"""
Demonstration of the pandas.io.clipboard Klipper assertion bug.
This code simulates what happens inside paste_klipper() function.
"""

def paste_klipper_mock(stdout_bytes):
    """Mock of the actual paste_klipper implementation from pandas.io.clipboard"""
    ENCODING = 'utf-8'
    clipboardContents = stdout_bytes.decode(ENCODING)

    # These assertions are from lines 277-279 of pandas/io/clipboard/__init__.py
    assert len(clipboardContents) > 0
    assert clipboardContents.endswith("\n")

    if clipboardContents.endswith("\n"):
        clipboardContents = clipboardContents[:-1]
    return clipboardContents

# Test case 1: Empty clipboard
print("Test 1: Empty clipboard (b'')")
try:
    result = paste_klipper_mock(b"")
    print(f"  Success: {repr(result)}")
except AssertionError:
    print(f"  AssertionError raised!")
    import traceback
    traceback.print_exc()

print("\nTest 2: Text without trailing newline (b'Hello')")
try:
    result = paste_klipper_mock(b"Hello")
    print(f"  Success: {repr(result)}")
except AssertionError:
    print(f"  AssertionError raised!")
    import traceback
    traceback.print_exc()

print("\nTest 3: Text with trailing newline (b'Hello\\n')")
try:
    result = paste_klipper_mock(b"Hello\n")
    print(f"  Success: {repr(result)}")
except AssertionError:
    print(f"  AssertionError raised!")
    import traceback
    traceback.print_exc()
