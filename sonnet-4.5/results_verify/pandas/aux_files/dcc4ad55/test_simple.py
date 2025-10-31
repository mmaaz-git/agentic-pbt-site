import tempfile
import os
from pandas.io.sas import read_sas

print("Testing file.xpt0:")
with tempfile.NamedTemporaryFile(delete=False, suffix='.file.xpt0') as tmp:
    tmp_path = tmp.name
    print(f"Created temp file: {tmp_path}")

try:
    read_sas(tmp_path)
except ValueError as e:
    print(f"Error: {e}")
    if "Header record" in str(e):
        print("BUG CONFIRMED: File was incorrectly detected as xport format")
    elif "unable to infer format" in str(e).lower():
        print("Correct behavior: Unable to infer format")
finally:
    os.unlink(tmp_path)

print("\nTesting data.xpt.backup:")
with tempfile.NamedTemporaryFile(delete=False, suffix='.data.xpt.backup') as tmp:
    tmp_path = tmp.name
    print(f"Created temp file: {tmp_path}")

try:
    read_sas(tmp_path)
except ValueError as e:
    print(f"Error: {e}")
    if "Header record" in str(e):
        print("BUG CONFIRMED: File was incorrectly detected as xport format")
    elif "unable to infer format" in str(e).lower():
        print("Correct behavior: Unable to infer format")
finally:
    os.unlink(tmp_path)

print("\nTesting myxptfile.txt:")
with tempfile.NamedTemporaryFile(delete=False, suffix='.myxptfile.txt') as tmp:
    tmp_path = tmp.name
    print(f"Created temp file: {tmp_path}")

try:
    read_sas(tmp_path)
except ValueError as e:
    print(f"Error: {e}")
    if "Header record" in str(e):
        print("BUG CONFIRMED: File was incorrectly detected as xport format")
    elif "unable to infer format" in str(e).lower():
        print("Correct behavior: Unable to infer format")
finally:
    os.unlink(tmp_path)

print("\nTesting valid.xpt:")
with tempfile.NamedTemporaryFile(delete=False, suffix='.valid.xpt') as tmp:
    tmp_path = tmp.name
    print(f"Created temp file: {tmp_path}")

try:
    read_sas(tmp_path)
except ValueError as e:
    print(f"Error: {e}")
    if "Header record" in str(e):
        print("Expected: File correctly detected as xport format (but not a valid xport file)")
    elif "unable to infer format" in str(e).lower():
        print("Unexpected behavior")
finally:
    os.unlink(tmp_path)