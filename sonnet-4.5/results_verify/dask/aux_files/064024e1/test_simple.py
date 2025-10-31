import pandas as pd
from dask.dataframe import from_pandas

def test_specific_case():
    """Test the specific failing case from the bug report"""
    df = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=20, freq='30min'),
        'value': range(20)
    })
    df = df.set_index('time')

    ddf = from_pandas(df, npartitions=2)

    # Test with window='1h', center=True
    print("Testing window='1h', center=True...")

    # Get pandas result
    pandas_result = df.rolling(window='1h', center=True).mean()
    print("Pandas result computed successfully")

    # Try dask result
    try:
        dask_result = ddf.rolling(window='1h', center=True).mean()
        dask_computed = dask_result.compute()
        print("Dask result computed successfully")
    except TypeError as e:
        print(f"Dask failed with TypeError: {e}")
        return False

    return True

# Test various combinations
test_cases = [
    ('1h', True, 2),
    ('2h', True, 3),
    ('30min', True, 1),
    ('1D', True, 4),
    ('1h', False, 2),  # This should work
    ('2h', False, 3),  # This should work
]

for window, center, npartitions in test_cases:
    df = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=20, freq='30min'),
        'value': range(20)
    })
    df = df.set_index('time')
    ddf = from_pandas(df, npartitions=npartitions)

    print(f"\nTesting: window={window}, center={center}, npartitions={npartitions}")
    try:
        result = ddf.rolling(window=window, center=center).mean()
        computed = result.compute()
        print(f"  ✓ SUCCESS")
    except TypeError as e:
        if "unsupported operand type(s) for //: 'str' and 'int'" in str(e):
            print(f"  ✗ FAIL (Expected bug): {e}")
        else:
            print(f"  ✗ FAIL (Unexpected error): {e}")