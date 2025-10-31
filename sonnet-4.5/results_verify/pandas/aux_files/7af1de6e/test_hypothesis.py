import pandas as pd
import json
import io
from hypothesis import given, strategies as st, settings


@given(st.lists(st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(max_size=50), st.none())
), min_size=1, max_size=50))
@settings(max_examples=100)
def test_jsonreader_basic_parsing(data):
    json_str = json.dumps(data)
    json_bytes = json_str.encode('utf-8')
    json_io = io.BytesIO(json_bytes)

    reader = pd.read_json(json_io, lines=False)

    assert len(reader) == len(data)

# Run the test
if __name__ == "__main__":
    # First test with the specific failing case
    print("Testing with specific failing case:")
    data = [{'0': -9_223_372_036_854_775_809}]

    try:
        json_str = json.dumps(data)
        json_bytes = json_str.encode('utf-8')
        json_io = io.BytesIO(json_bytes)
        reader = pd.read_json(json_io, lines=False)
        print(f"Success! Length check: {len(reader)} == {len(data)}")
    except Exception as e:
        print(f"Failed with error: {type(e).__name__}: {e}")

    print("\n" + "="*50 + "\n")
    print("Running hypothesis tests (100 examples)...")

    try:
        test_jsonreader_basic_parsing()
        print("All tests passed!")
    except Exception as e:
        print(f"Hypothesis test failed: {e}")