#!/usr/bin/env python3

from io import StringIO
import pandas as pd

def test_specific_roundtrip():
    data = [[1.7976931345e+308]]

    df = pd.DataFrame(data)
    json_str = df.to_json(orient='columns')
    df_recovered = pd.read_json(StringIO(json_str), orient='columns')

    print(f"Original: {df.iloc[0, 0]}")
    print(f"Recovered: {df_recovered.iloc[0, 0]}")
    print(f"Are they equal: {df.iloc[0, 0] == df_recovered.iloc[0, 0]}")

    try:
        pd.testing.assert_frame_equal(df, df_recovered, check_dtype=False)
        print("Test PASSED")
    except AssertionError as e:
        print(f"Test FAILED: {e}")

test_specific_roundtrip()