#!/usr/bin/env python3
"""Test script to reproduce the _split_line bug"""

print("Testing pandas.io.sas.sas_xport._split_line bug")
print("-" * 50)

# First, test the basic reproduction case
try:
    from pandas.io.sas.sas_xport import _split_line

    print("\nTest 1: Basic reproduction without '_' field")
    parts = [("name", 10), ("value", 10)]
    test_string = "hello     world     "

    try:
        result = _split_line(test_string, parts)
        print(f"Result: {result}")
    except KeyError as e:
        print(f"KeyError raised as expected: {e}")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")

    print("\nTest 2: With '_' field (expected usage)")
    parts = [("name", 10), ("_", 5), ("value", 5)]
    test_string = "hello     xxx  world"

    try:
        result = _split_line(test_string, parts)
        print(f"Result: {result}")
        print("Success - no error when '_' is present")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

    # Test the property-based test
    print("\nTest 3: Property-based test with hypothesis")
    try:
        from hypothesis import given, strategies as st, assume

        @given(
            st.lists(
                st.tuples(
                    st.text(min_size=1, max_size=10),
                    st.integers(min_value=1, max_value=20)
                ),
                min_size=1,
                max_size=10
            )
        )
        def test_split_line_without_underscore(parts):
            assume("_" not in [name for name, _ in parts])

            total_len = sum(length for _, length in parts)
            test_string = "x" * total_len

            result = _split_line(test_string, parts)
            assert len(result) == len(parts)

        # Run the hypothesis test
        test_split_line_without_underscore()
        print("Hypothesis test passed - no failures found")

    except ImportError:
        print("Hypothesis not installed, skipping property-based test")
    except AssertionError as e:
        print(f"Hypothesis test failed as expected: {e}")
    except KeyError as e:
        print(f"Hypothesis test found KeyError as expected: {e}")
    except Exception as e:
        print(f"Hypothesis test error: {type(e).__name__}: {e}")

except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

print("\n" + "-" * 50)
print("Testing complete")