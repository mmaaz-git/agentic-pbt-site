#!/usr/bin/env python3
"""Test to reproduce the reported bug in sorted_columns function"""

import sys
import traceback

# First, let's test the exact reproduction case from the bug report
print("=" * 60)
print("Testing Bug Reproduction Case")
print("=" * 60)

try:
    import dask.dataframe.io.parquet.core as parquet_core

    statistics = [
        {'columns': [
            {'name': 'col1', 'min': 0, 'max': 10},
            {'name': 'col2', 'min': 0, 'max': 10}
        ]},
        {'columns': [
            {'name': 'col1', 'min': 11, 'max': 20}
        ]}
    ]

    print("Input statistics:")
    for i, stat in enumerate(statistics):
        print(f"  Row group {i}: {len(stat['columns'])} columns")
        for col in stat['columns']:
            print(f"    - {col['name']}: min={col['min']}, max={col['max']}")

    print("\nCalling sorted_columns(statistics)...")
    result = parquet_core.sorted_columns(statistics)
    print(f"Result: {result}")
    print("No error occurred - bug may not be present or has been fixed")

except IndexError as e:
    print(f"IndexError occurred as reported: {e}")
    print(f"Traceback:")
    traceback.print_exc()

except Exception as e:
    print(f"Unexpected error: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Testing Property-Based Test")
print("=" * 60)

# Now test with the hypothesis property test
try:
    from hypothesis import given, strategies as st, settings
    import dask.dataframe.io.parquet.core as parquet_core

    @given(st.lists(
        st.fixed_dictionaries({
            'columns': st.lists(
                st.fixed_dictionaries({
                    'name': st.text(min_size=1, max_size=20),
                    'min': st.one_of(st.none(), st.integers(-1000, 1000)),
                    'max': st.one_of(st.none(), st.integers(-1000, 1000))
                }),
                min_size=1,
                max_size=5
            )
        }),
        min_size=1,
        max_size=10
    ))
    @settings(max_examples=100, deadline=None)
    def test_sorted_columns_divisions_are_sorted(statistics):
        try:
            result = parquet_core.sorted_columns(statistics)
            for item in result:
                assert item['divisions'] == sorted(item['divisions'])
        except IndexError:
            # Track if IndexError occurs
            return False
        return True

    print("Running property-based test with 100 examples...")
    errors_found = 0

    # Run the test manually to catch errors
    test_func = test_sorted_columns_divisions_are_sorted
    for i in range(100):
        try:
            test_func()
        except AssertionError as e:
            if "IndexError" in str(e):
                errors_found += 1
        except Exception as e:
            if isinstance(e, IndexError):
                errors_found += 1

    if errors_found > 0:
        print(f"Found {errors_found} IndexError cases in property test")
    else:
        print("No IndexErrors found in property test")

except ImportError:
    print("hypothesis not installed, skipping property test")
except Exception as e:
    print(f"Error in property test: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Testing Edge Cases")
print("=" * 60)

# Test various edge cases
test_cases = [
    # Case 1: Empty statistics
    {
        'name': 'Empty statistics',
        'input': [],
        'should_fail': False
    },
    # Case 2: Single row group
    {
        'name': 'Single row group',
        'input': [
            {'columns': [
                {'name': 'col1', 'min': 0, 'max': 10}
            ]}
        ],
        'should_fail': False
    },
    # Case 3: Mismatched columns (different names)
    {
        'name': 'Different column names',
        'input': [
            {'columns': [
                {'name': 'col1', 'min': 0, 'max': 10}
            ]},
            {'columns': [
                {'name': 'col2', 'min': 11, 'max': 20}
            ]}
        ],
        'should_fail': True  # Should fail due to IndexError
    },
    # Case 4: More columns in second row group
    {
        'name': 'More columns in second row group',
        'input': [
            {'columns': [
                {'name': 'col1', 'min': 0, 'max': 10}
            ]},
            {'columns': [
                {'name': 'col1', 'min': 11, 'max': 20},
                {'name': 'col2', 'min': 0, 'max': 10}
            ]}
        ],
        'should_fail': False  # Should not fail - can access index 0
    },
    # Case 5: Missing min/max values
    {
        'name': 'Missing min/max values',
        'input': [
            {'columns': [
                {'name': 'col1', 'min': None, 'max': None}
            ]},
            {'columns': [
                {'name': 'col1', 'min': 11, 'max': 20}
            ]}
        ],
        'should_fail': False  # Should handle None values
    }
]

for test_case in test_cases:
    print(f"\nTest: {test_case['name']}")
    print(f"Input: {test_case['input']}")
    try:
        result = parquet_core.sorted_columns(test_case['input'])
        print(f"Result: {result}")
        if test_case['should_fail']:
            print("  WARNING: Expected IndexError but succeeded")
    except IndexError as e:
        print(f"  IndexError: {e}")
        if not test_case['should_fail']:
            print("  WARNING: Unexpected IndexError")
    except Exception as e:
        print(f"  Other error: {e}")