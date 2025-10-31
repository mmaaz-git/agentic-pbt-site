import tempfile

print("Testing double close() on regular Python file objects...")

# Test regular file
with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
    tmp_path = tmp.name

try:
    f = open(tmp_path, 'w')
    f.write("test")
    print("First file.close()...")
    f.close()
    print("First close() succeeded")

    print("Second file.close()...")
    f.close()  # Should not raise
    print("Second close() succeeded - no error!")
except Exception as e:
    print(f"Error on second close(): {type(e).__name__}: {e}")
finally:
    import os
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)