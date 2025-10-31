def paste_klipper_mock(stdout_bytes):
    ENCODING = 'utf-8'
    clipboardContents = stdout_bytes.decode(ENCODING)

    # These assertions can fail
    assert len(clipboardContents) > 0  # Fails on empty clipboard
    assert clipboardContents.endswith("\n")  # Fails if no trailing newline

    if clipboardContents.endswith("\n"):
        clipboardContents = clipboardContents[:-1]
    return clipboardContents

# Test cases that cause assertion failures:
print("Testing empty clipboard...")
try:
    paste_klipper_mock(b"")  # Empty clipboard
    print("ERROR: Should have raised AssertionError for empty clipboard")
except AssertionError as e:
    print(f"✓ Crash on empty clipboard (expected): {e}")

print("\nTesting data without newline...")
try:
    paste_klipper_mock(b"Hello")  # No trailing newline
    print("ERROR: Should have raised AssertionError for data without newline")
except AssertionError as e:
    print(f"✓ Crash on data without newline (expected): {e}")

print("\nTesting valid data with newline...")
try:
    result = paste_klipper_mock(b"Hello\n")  # Valid data with newline
    print(f"✓ Valid data works: '{result}'")
except AssertionError as e:
    print(f"ERROR: Should not have crashed on valid data: {e}")