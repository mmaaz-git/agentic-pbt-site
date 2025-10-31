#!/usr/bin/env python3
"""Analyze the sorted_columns function behavior"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

def analyze_sorted_columns():
    """Analyze what sorted_columns is supposed to do based on the code"""

    print("Code analysis of sorted_columns function:")
    print("=" * 60)
    print()

    print("Function purpose (from docstring):")
    print("- Find sorted columns given row-group statistics")
    print("- Returns columns that are sorted with their divisions")
    print("- Divisions are boundaries for data partitioning")
    print()

    print("Key logic flow:")
    print("1. Line 421-424: Skip columns that don't have both min AND max in all statistics")
    print("2. Line 425-427: Initialize with first row group's min/max, success=True if min is not None")
    print("3. Line 428-438: For subsequent row groups:")
    print("   - Line 430-432: Fail if min is None")
    print("   - Line 433-435: If min >= max (from previous), add to divisions")
    print("   - Line 436-438: Otherwise fail (not sorted)")
    print("4. Line 440-443: If successful, add final max and verify sorting")
    print()

    print("The problematic scenario:")
    print("- Input: [{'columns': [{'name': '0', 'min': 0, 'max': None}]}]")
    print("- Line 421-424: Checks 'min' in columns[0] AND 'max' in columns[0]")
    print("  Both keys exist (even though max=None), so check passes")
    print("- Line 425-426: divisions = [0], max = None")
    print("- Line 427: success = (0 is not None) = True")
    print("- Line 440-443: Since success=True and there's only 1 row group:")
    print("  - Line 441: divisions.append(None) -> divisions = [0, None]")
    print("  - Line 442: sorted([0, None]) -> TypeError!")
    print()

    print("The key issue:")
    print("- Line 421-424 checks if 'min' and 'max' keys EXIST")
    print("- It doesn't check if their VALUES are not None")
    print("- Line 427 only checks if min is not None")
    print("- Line 442 assumes all values in divisions are comparable")

if __name__ == "__main__":
    analyze_sorted_columns()