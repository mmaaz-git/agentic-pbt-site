from pandas.io.formats.format import format_percentiles

percentiles = [0.0, 7.506590166045388e-253]
formatted = format_percentiles(percentiles)

print(f"Input: {percentiles}")
print(f"Output: {formatted}")
print(f"Are outputs unique? {len(formatted) == len(set(formatted))}")

try:
    assert len(formatted) == len(set(formatted)), "Unique inputs should produce unique outputs"
    print("Test passed!")
except AssertionError as e:
    print(f"AssertionError: {e}")