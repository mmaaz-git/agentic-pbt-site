from hypothesis import given, settings, strategies as st, example
import pandas as pd
from io import StringIO

@given(
    st.floats(
        allow_nan=False,
        allow_infinity=False,
        min_value=-1e10,
        max_value=1e10,
        width=64
    )
)
@settings(max_examples=500)
@example(1.5932223682757467)  # Add the specific failing example
def test_dataframe_json_roundtrip(value):
    df = pd.DataFrame({'col': [value]})
    json_str = df.to_json(orient='records')
    df_restored = pd.read_json(StringIO(json_str), orient='records')

    orig = df['col'].iloc[0]
    restored = df_restored['col'].iloc[0]

    assert orig == restored, f"Round-trip failed: {orig} != {restored}"

# Test with the specific failing value mentioned
if __name__ == "__main__":
    print("Testing with specific value: 1.5932223682757467")
    value = 1.5932223682757467
    df = pd.DataFrame({'col': [value]})
    json_str = df.to_json(orient='records')
    df_restored = pd.read_json(StringIO(json_str), orient='records')

    orig = df['col'].iloc[0]
    restored = df_restored['col'].iloc[0]

    print(f"Original:  {orig:.17f}")
    print(f"JSON:      {json_str}")
    print(f"Restored:  {restored:.17f}")
    print(f"Match:     {orig == restored}")

    if orig != restored:
        print(f"Test failed: Round-trip failed: {orig} != {restored}")
    else:
        print("Test passed!")

    print("\nRunning hypothesis tests...")
    test_dataframe_json_roundtrip()