#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.io.sas.sasreader import read_sas

print("Testing format detection logic:")
print("-" * 60)

filenames = [
    "test.xpt.sas7bdat",
    "data.sas7bdat.xpt",
    "archive.xpt.backup",
    "my.xpt_notes.txt",
    "data.sas7bdat.old",
]

for filename in filenames:
    fname_lower = filename.lower()
    if ".xpt" in fname_lower:
        detected = "xport"
    elif ".sas7bdat" in fname_lower:
        detected = "sas7bdat"
    else:
        detected = "unknown"

    print(f"{filename:25s} -> {detected}")

print("\n" + "=" * 60)
print("Testing actual read_sas function behavior:")
print("-" * 60)

# Test what read_sas actually does
for filename in filenames:
    try:
        # This should fail since the files don't exist, but we'll see what error we get
        read_sas(filename)
    except FileNotFoundError as e:
        print(f"{filename:25s} -> FileNotFoundError (format was accepted)")
    except ValueError as e:
        if "unable to infer format" in str(e):
            print(f"{filename:25s} -> ValueError: unable to infer format")
        else:
            print(f"{filename:25s} -> ValueError: {e}")
    except Exception as e:
        print(f"{filename:25s} -> {type(e).__name__}: {e}")