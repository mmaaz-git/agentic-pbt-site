import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.utils import format_bytes
from hypothesis import given, strategies as st, settings, example

violations_found = []

@settings(max_examples=1000)
@given(st.integers(min_value=0, max_value=2**60-1))
@example(1000 * 2**50)  # Specifically test the problematic case
def test_format_bytes_length_constraint_documented(n):
    """Property: format_bytes should output <= 10 chars for n < 2**60 (documented claim)"""
    result = format_bytes(n)

    if n < 2**60:
        if len(result) > 10:
            violations_found.append((n, result, len(result)))

# Run the test
try:
    test_format_bytes_length_constraint_documented()
    if violations_found:
        print(f"Found {len(violations_found)} violations:")
        for n, result, length in violations_found[:10]:  # Show first 10
            print(f"  format_bytes({n}) = '{result}' (length={length})")
    else:
        print("No violations found")
except Exception as e:
    print(f"Test error: {e}")
    if violations_found:
        print(f"But found {len(violations_found)} violations before error:")
        for n, result, length in violations_found[:10]:
            print(f"  format_bytes({n}) = '{result}' (length={length})")