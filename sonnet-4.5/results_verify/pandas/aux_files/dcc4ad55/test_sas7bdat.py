import tempfile
import os
from pandas.io.sas import read_sas

print("Testing file.sas7bdat0:")
with tempfile.NamedTemporaryFile(delete=False, suffix='.file.sas7bdat0') as tmp:
    tmp_path = tmp.name
    print(f"Created temp file: {tmp_path}")

try:
    read_sas(tmp_path)
except Exception as e:
    print(f"Error: {e}")
    if "magic number mismatch" in str(e).lower():
        print("BUG CONFIRMED: File was incorrectly detected as sas7bdat format")
    elif "unable to infer format" in str(e).lower():
        print("Correct behavior: Unable to infer format")
finally:
    os.unlink(tmp_path)

print("\nTesting data.sas7bdat.backup:")
with tempfile.NamedTemporaryFile(delete=False, suffix='.data.sas7bdat.backup') as tmp:
    tmp_path = tmp.name
    print(f"Created temp file: {tmp_path}")

try:
    read_sas(tmp_path)
except Exception as e:
    print(f"Error: {e}")
    if "magic number mismatch" in str(e).lower():
        print("BUG CONFIRMED: File was incorrectly detected as sas7bdat format")
    elif "unable to infer format" in str(e).lower():
        print("Correct behavior: Unable to infer format")
finally:
    os.unlink(tmp_path)