import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings

# Test with explicit ß character
def test_explicit_eszett():
    strings = ['ß', 'Straße', 'groß', 'weiß']
    pdf = pd.DataFrame({'text': strings})
    ddf = dd.from_pandas(pdf, npartitions=2)

    pandas_result = pdf['text'].str.upper()
    dask_result = ddf['text'].str.upper().compute()

    print("Testing strings with ß character:")
    for i in range(len(strings)):
        print(f"  Input: '{strings[i]}'")
        print(f"  Pandas: '{pandas_result.iloc[i]}'")
        print(f"  Dask:   '{dask_result.iloc[i]}'")
        if pandas_result.iloc[i] != dask_result.iloc[i]:
            print(f"  MISMATCH!")
        else:
            print(f"  Match")
        print()

if __name__ == "__main__":
    test_explicit_eszett()