#!/usr/bin/env python3
"""Test the reported bug in dask.dataframe.io.parquet.core.sorted_columns"""

# First, let's test the exact reproducer from the bug report
print("=" * 60)
print("Testing exact reproducer from bug report")
print("=" * 60)

from dask.dataframe.io.parquet.core import sorted_columns

statistics = [{"columns": [{"name": "a", "min": 0, "max": None}]}]

try:
    result = sorted_columns(statistics)
    print(f"Result: {result}")
    print("No error occurred - bug may not exist")
except TypeError as e:
    print(f"TypeError occurred: {e}")
    print("Bug confirmed - TypeError when comparing None with integers")
except AssertionError as e:
    print(f"AssertionError occurred: {e}")
    print("Bug confirmed - assertion fails when divisions contain None")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

# Now let's test the property-based test
print("\n" + "=" * 60)
print("Testing with Hypothesis property-based test")
print("=" * 60)

from hypothesis import given, strategies as st, settings
import string

@st.composite
def statistics_strategy(draw):
    num_row_groups = draw(st.integers(min_value=0, max_value=20))
    num_columns = draw(st.integers(min_value=1, max_value=5))

    column_names = [
        draw(st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=10))
        for _ in range(num_columns)
    ]

    stats = []
    for _ in range(num_row_groups):
        columns = []
        for col_name in column_names:
            has_stats = draw(st.booleans())
            if has_stats:
                min_val = draw(st.integers(min_value=-1000, max_value=1000) | st.none())
                if min_val is not None:
                    max_val = draw(
                        st.integers(min_value=min_val, max_value=1000) | st.none()
                    )
                else:
                    max_val = None
                columns.append({"name": col_name, "min": min_val, "max": max_val})
            else:
                columns.append({"name": col_name})

        stats.append({"columns": columns})

    return stats


@given(statistics_strategy())
@settings(max_examples=100)  # Reduced from 1000 for faster testing
def test_sorted_columns_divisions_are_sorted(statistics):
    result = sorted_columns(statistics)
    for col_info in result:
        divisions = col_info["divisions"]
        assert divisions == sorted(divisions)

# Run the test and collect failures
failures = []
print("Running property-based tests...")
try:
    test_sorted_columns_divisions_are_sorted()
    print("All property tests passed")
except Exception as e:
    print(f"Property test failed: {e}")
    failures.append(e)

# Test some edge cases
print("\n" + "=" * 60)
print("Testing additional edge cases")
print("=" * 60)

test_cases = [
    # Case 1: None max value (from bug report)
    {"name": "None max", "stats": [{"columns": [{"name": "a", "min": 0, "max": None}]}]},

    # Case 2: None min value
    {"name": "None min", "stats": [{"columns": [{"name": "a", "min": None, "max": 10}]}]},

    # Case 3: Both None
    {"name": "Both None", "stats": [{"columns": [{"name": "a", "min": None, "max": None}]}]},

    # Case 4: Multiple row groups with None
    {"name": "Multiple with None", "stats": [
        {"columns": [{"name": "a", "min": 0, "max": 5}]},
        {"columns": [{"name": "a", "min": 6, "max": None}]}
    ]},

    # Case 5: Normal case (should work)
    {"name": "Normal case", "stats": [
        {"columns": [{"name": "a", "min": 0, "max": 5}]},
        {"columns": [{"name": "a", "min": 6, "max": 10}]}
    ]}
]

for test_case in test_cases:
    print(f"\nTest: {test_case['name']}")
    print(f"Input: {test_case['stats']}")
    try:
        result = sorted_columns(test_case['stats'])
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")