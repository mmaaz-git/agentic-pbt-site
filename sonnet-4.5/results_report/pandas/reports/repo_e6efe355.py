import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.io.sas.sasreader import read_sas

print("Testing SAS format detection bug with substring matching\n")
print("=" * 60)

test_cases = [
    ("archive.xpt.backup", "Should reject as .backup file"),
    ("data.sas7bdat.old", "Should reject as .old file"),
    ("my.xpt_notes.txt", "Should reject as .txt file"),
    ("test.xpt.sas7bdat", "Ambiguous: contains both substrings"),
    ("data.sas7bdat.xpt", "Ambiguous: contains both substrings"),
    (".xpt", "Edge case: no filename, just extension"),
    (".sas7bdat", "Edge case: no filename, just extension"),
    ("test.txt", "Should reject: no SAS substring"),
]

for filename, description in test_cases:
    print(f"\nTest: {filename}")
    print(f"Description: {description}")
    print("-" * 40)

    try:
        # Try to read the file (will fail due to file not existing, but we'll see format detection)
        reader = read_sas(filename)
    except ValueError as e:
        if "unable to infer format" in str(e):
            print(f"Result: ValueError - unable to infer format")
            print(f"Error: {e}")
        else:
            print(f"Result: ValueError (other)")
            print(f"Error: {e}")
    except FileNotFoundError as e:
        # This means format was detected and it tried to open the file
        fname_lower = filename.lower()
        if ".xpt" in fname_lower:
            detected = "xport"
        elif ".sas7bdat" in fname_lower:
            detected = "sas7bdat"
        else:
            detected = "unknown"
        print(f"Result: Format detected as '{detected}' (FileNotFoundError when opening)")
        print(f"Error: {e}")
    except Exception as e:
        print(f"Result: Unexpected error")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")

print("\n" + "=" * 60)
print("\nSUMMARY: The bug allows files with '.xpt' or '.sas7bdat' anywhere")
print("in their filename to be incorrectly accepted as valid SAS files,")
print("regardless of their actual file extension.")