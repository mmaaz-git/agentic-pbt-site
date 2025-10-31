from pandas.io.sas import read_sas

# Test 1: File with .xpt embedded but not as extension
print("Test 1: Filename 'data.xpt.backup'")
try:
    read_sas("data.xpt.backup")
except FileNotFoundError as e:
    print(f"  FileNotFoundError: {e}")
    print("  BUG: File was incorrectly detected as xport format")
except ValueError as e:
    if "unable to infer format" in str(e):
        print(f"  ValueError: {e}")
        print("  CORRECT: Format detection failed appropriately")
    else:
        print(f"  ValueError: {e}")
        print("  BUG: Unexpected error")

print("\nTest 2: Filename 'file.sas7bdat.old'")
try:
    read_sas("file.sas7bdat.old")
except FileNotFoundError as e:
    print(f"  FileNotFoundError: {e}")
    print("  BUG: File was incorrectly detected as sas7bdat format")
except ValueError as e:
    if "unable to infer format" in str(e):
        print(f"  ValueError: {e}")
        print("  CORRECT: Format detection failed appropriately")
    else:
        print(f"  ValueError: {e}")
        print("  BUG: Unexpected error")

print("\nTest 3: Filename 'myfile.xpt123'")
try:
    read_sas("myfile.xpt123")
except FileNotFoundError as e:
    print(f"  FileNotFoundError: {e}")
    print("  BUG: File was incorrectly detected as xport format")
except ValueError as e:
    if "unable to infer format" in str(e):
        print(f"  ValueError: {e}")
        print("  CORRECT: Format detection failed appropriately")
    else:
        print(f"  ValueError: {e}")
        print("  BUG: Unexpected error")