from pandas.io.formats.format import format_percentiles

# Test case with extremely small value close to zero
percentiles = [0.0, 7.506590166045388e-253]

# Format the percentiles
formatted = format_percentiles(percentiles)

# Display results
print(f"Input percentiles: {percentiles}")
print(f"Unique input count: {len(set(percentiles))}")
print(f"Output formatted: {formatted}")
print(f"Unique output count: {len(set(formatted))}")

# Check if uniqueness is preserved
try:
    assert len(formatted) == len(set(formatted)), \
        f"Unique inputs should produce unique outputs, but got {formatted}"
    print("\nAssertion passed: Uniqueness preserved")
except AssertionError as e:
    print(f"\nAssertion failed: {e}")