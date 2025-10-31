"""Detailed trace of the bug to understand the issue precisely"""

from dask.dataframe.io.parquet.core import sorted_columns

print("Tracing through the issue step by step:")

stats = [
    {"columns": [{"name": "a", "min": None, "max": None}]},
    {"columns": [{"name": "a", "min": 5, "max": 10}]},
]

print(f"Input statistics: {stats}")
print("\n--- Starting sorted_columns execution ---")

# Manually trace through the function logic
if not stats:
    print("Statistics is empty - would return []")
else:
    print("Statistics not empty, continuing...")

out = []
for i, c in enumerate(stats[0]["columns"]):
    print(f"\nProcessing column {i}: {c}")

    # Check if all statistics have min and max
    has_all_minmax = all(
        "min" in s["columns"][i] and "max" in s["columns"][i] for s in stats
    )
    print(f"  All row groups have min/max keys: {has_all_minmax}")

    if not has_all_minmax:
        print("  Skipping column due to missing min/max keys")
        continue

    # Initialize divisions
    divisions = [c["min"]]
    max_val = c["max"]
    success = c["min"] is not None

    print(f"  Initial divisions: {divisions}")
    print(f"  Initial max: {max_val}")
    print(f"  Initial success: {success}")

    # Process remaining statistics
    for j, stat_entry in enumerate(stats[1:], 1):
        col = stat_entry["columns"][i]
        print(f"\n  Processing row group {j}: {col}")
        print(f"    Checking if col['min'] ({col['min']}) is None...")

        if col["min"] is None:
            print("    col['min'] is None, setting success=False and breaking")
            success = False
            break

        print(f"    col['min'] is not None")
        print(f"    Comparing col['min'] ({col['min']}) >= max ({max_val})...")

        # THIS IS WHERE THE BUG OCCURS!
        try:
            comparison = col["min"] >= max_val
            print(f"    Comparison result: {comparison}")
        except TypeError as e:
            print(f"    ERROR in comparison: {e}")
            print(f"    This happens because max_val is {max_val} and col['min'] is {col['min']}")
            print("    The function doesn't check if max_val is None before the comparison!")
            raise