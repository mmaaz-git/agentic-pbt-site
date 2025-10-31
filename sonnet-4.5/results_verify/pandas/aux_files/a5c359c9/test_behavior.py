from io import BytesIO

# Simulate what happens in inspect_excel_format
stream = BytesIO(b'')
stream.seek(0)
buf = stream.read(8)  # PEEK_SIZE is 8 based on the code

print(f"buf value: {repr(buf)}")
print(f"buf is None: {buf is None}")
print(f"not buf: {not buf}")
print(f"len(buf): {len(buf)}")
print(f"bool(buf): {bool(buf)}")

# The bug: Code checks if buf is None, but empty bytes b'' is not None
if buf is None:
    print("Would raise ValueError (current code)")
else:
    print("Would NOT raise ValueError (current code)")

# The fix: Check if buf is empty
if not buf:
    print("Would raise ValueError (fixed code)")
else:
    print("Would NOT raise ValueError (fixed code)")