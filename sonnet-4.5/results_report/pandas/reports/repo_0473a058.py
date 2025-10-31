import tempfile
import os
from pandas.io.sas import read_sas

# Test case: file.xpt0 - should NOT be detected as xport format
print("Testing file.xpt0:")
with tempfile.NamedTemporaryFile(delete=False, suffix='.file.xpt0') as tmp:
    tmp_path = tmp.name

try:
    read_sas(tmp_path)
except ValueError as e:
    print(f"Error: {e}")
    if "Header record" in str(e):
        print("BUG CONFIRMED: File was incorrectly detected as xport format!")
    elif "unable to infer format" in str(e):
        print("CORRECT: File was not detected as a SAS file")
finally:
    os.unlink(tmp_path)

print("\n" + "="*50 + "\n")

# Test case: data.xpt.backup - backup file should NOT be detected as xport
print("Testing data.xpt.backup:")
with tempfile.NamedTemporaryFile(delete=False, suffix='.data.xpt.backup') as tmp:
    tmp_path = tmp.name

try:
    read_sas(tmp_path)
except ValueError as e:
    print(f"Error: {e}")
    if "Header record" in str(e):
        print("BUG CONFIRMED: Backup file was incorrectly detected as xport format!")
    elif "unable to infer format" in str(e):
        print("CORRECT: File was not detected as a SAS file")
finally:
    os.unlink(tmp_path)

print("\n" + "="*50 + "\n")

# Test case: file.sas7bdat0 - should NOT be detected as sas7bdat format
print("Testing file.sas7bdat0:")
with tempfile.NamedTemporaryFile(delete=False, suffix='.file.sas7bdat0') as tmp:
    tmp_path = tmp.name

try:
    read_sas(tmp_path)
except ValueError as e:
    print(f"Error: {e}")
    if "magic number" in str(e):
        print("BUG CONFIRMED: File was incorrectly detected as sas7bdat format!")
    elif "unable to infer format" in str(e):
        print("CORRECT: File was not detected as a SAS file")
finally:
    os.unlink(tmp_path)