import pandas as pd
import sys

print(f"Pandas version: {pd.__version__}")
print()

print("Testing pandas behavior with timestamps outside ns range...")
print("=" * 60)

# Test creating timestamps
test_years = [1, 1000, 1677, 1678, 2262, 2263, 9999]

for year in test_years:
    print(f"\nYear {year}:")

    # Create timestamp
    try:
        ts = pd.Timestamp(year=year, month=1, day=1)
        print(f"  Created: {ts}")
        print(f"  Unit: {ts.unit}")
        print(f"  Has as_unit: {hasattr(ts, 'as_unit')}")
        print(f"  Has _as_unit: {hasattr(ts, '_as_unit')}")

        # Test as_unit if available
        if hasattr(ts, 'as_unit'):
            for unit in ['s', 'ms', 'us', 'ns']:
                try:
                    result = ts.as_unit(unit)
                    print(f"  as_unit('{unit}'): Success - unit={result.unit}")
                except Exception as e:
                    print(f"  as_unit('{unit}'): {type(e).__name__}")
        elif hasattr(ts, '_as_unit'):
            for unit in ['s', 'ms', 'us', 'ns']:
                try:
                    result = ts._as_unit(unit)
                    print(f"  _as_unit('{unit}'): Success - unit={result.unit}")
                except Exception as e:
                    print(f"  _as_unit('{unit}'): {type(e).__name__}")

    except Exception as e:
        print(f"  ERROR creating timestamp: {e}")

print()
print("=" * 60)
print("Testing pd.Timestamp with explicit unit parameter...")
print()

# Test if we can create timestamps with explicit units
for year in [1, 1000, 9999]:
    for unit in ['s', 'ms', 'us', 'ns']:
        try:
            # Try creating with unit parameter if supported
            ts = pd.Timestamp(year=year, month=1, day=1, unit=unit)
            print(f"pd.Timestamp(year={year}, unit='{unit}'): Success - {ts}, unit={ts.unit}")
        except TypeError:
            # If unit parameter is not supported, create normally
            try:
                ts = pd.Timestamp(year=year, month=1, day=1)
                print(f"pd.Timestamp(year={year}): Success - {ts}, unit={ts.unit}")
            except Exception as e:
                print(f"pd.Timestamp(year={year}): {type(e).__name__}: {e}")
        except Exception as e:
            print(f"pd.Timestamp(year={year}, unit='{unit}'): {type(e).__name__}: {e}")