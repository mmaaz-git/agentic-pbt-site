from hypothesis import given, strategies as st
from dask.array.overlap import ensure_minimum_chunksize
import inspect

# First, run the hypothesis test
@given(
    min_size=st.integers(min_value=1, max_value=20),
    chunks=st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=10)
)
def test_ensure_minimum_chunksize_enforces_minimum(min_size, chunks):
    """Property: All output chunks should be >= min_size (except possibly the last)"""
    chunks_tuple = tuple(chunks)
    try:
        result = ensure_minimum_chunksize(min_size, chunks_tuple)
        # Verify the property: result chunks are >= min_size
        assert all(c >= min_size for c in result), \
            f"Function should ensure all chunks >= {min_size}, got {result}"
    except ValueError:
        # Expected when min_size > sum(chunks)
        assert min_size > sum(chunks)

# Run hypothesis test
print("Running hypothesis test...")
test_ensure_minimum_chunksize_enforces_minimum()
print("Hypothesis test passed!")

# Now run the reproduction code
print("\n" + "="*60 + "\n")

source = inspect.getsource(ensure_minimum_chunksize)
docstring = ensure_minimum_chunksize.__doc__

print("Function name: ensure_minimum_chunksize")
print("Function docstring (first line):")
if docstring:
    lines = docstring.strip().split('\n')
    if lines:
        print(f"  '{lines[0]}'")
print()

# Let's print the actual docstring to see what it says
print("Full docstring:")
print("-" * 40)
print(docstring)
print("-" * 40)
print()

result = ensure_minimum_chunksize(10, (20, 20, 1))
print(f"Test: ensure_minimum_chunksize(10, (20, 20, 1)) = {result}")
print(f"Expected (from docstring example): (20, 11, 10)")
print(f"Result: {result}")
print()
print("Analysis: All output chunks are >= 10, confirming MINIMUM enforcement")
print("The parameter doc should say 'minimum' not 'maximum'")

# Additional test cases
print("\n" + "="*60 + "\n")
print("Additional test cases:")
print(f"ensure_minimum_chunksize(5, (1, 2, 3)) = {ensure_minimum_chunksize(5, (1, 2, 3))}")
print(f"ensure_minimum_chunksize(10, (5, 5, 5)) = {ensure_minimum_chunksize(10, (5, 5, 5))}")
print(f"ensure_minimum_chunksize(3, (10, 10, 10)) = {ensure_minimum_chunksize(3, (10, 10, 10))}")