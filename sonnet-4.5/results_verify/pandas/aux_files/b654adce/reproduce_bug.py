#!/usr/bin/env python3
"""Reproduce the duplicate assignment bug in SAS7BDATReader.__init__"""

def test_duplicate_assignment():
    # Read the source file and check for duplicate assignment
    with open('/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/sas/sas7bdat.py') as f:
        lines = f.readlines()

    # Check lines 204-208 (0-indexed, so 204 becomes 203)
    print("Lines 204-208 from sas7bdat.py:")
    for i in range(204, 209):
        print(f"Line {i}: {lines[i].rstrip()}")

    # Check for the duplicate
    line_205 = lines[204].strip()
    line_207 = lines[206].strip()

    print(f"\nLine 205: {line_205}")
    print(f"Line 207: {line_207}")

    if line_205 == line_207 and line_205 == "self._current_row_in_file_index = 0":
        print("\n✓ CONFIRMED: Duplicate assignment found!")
        print("Both lines 205 and 207 contain: self._current_row_in_file_index = 0")
        return True
    else:
        print("\n✗ Bug NOT confirmed")
        return False

if __name__ == "__main__":
    test_duplicate_assignment()