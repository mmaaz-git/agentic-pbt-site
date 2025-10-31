from hypothesis import given, strategies as st
import pandas as pd
import io

@given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers()))
def test_json_roundtrip_preserves_length(data):
    df = pd.DataFrame([data])
    json_str = df.to_json(orient='records')
    df_back = pd.read_json(io.StringIO(json_str), orient='records')
    assert len(df_back) == len(df), f"Original: {len(df)}, Restored: {len(df_back)}, Data: {data}"

# Test with the failing input directly
def test_empty_dict():
    data = {}
    df = pd.DataFrame([data])
    json_str = df.to_json(orient='records')
    df_back = pd.read_json(io.StringIO(json_str), orient='records')
    assert len(df_back) == len(df), f"Original: {len(df)}, Restored: {len(df_back)}, Data: {data}"

print("Testing with empty dictionary:")
try:
    test_empty_dict()
    print("Test passed")
except AssertionError as e:
    print(f"Test failed: {e}")

# Run hypothesis tests
print("\nRunning Hypothesis tests:")
test_json_roundtrip_preserves_length()