import pandas as pd

df = pd.DataFrame({
    'time': pd.date_range('2020-01-01', periods=10, freq='1h'),
    'value': range(10)
})
df = df.set_index('time')

print("Testing pandas rolling with window='2h', center=True...")
try:
    result = df.rolling(window='2h', center=True).mean()
    print("Success! Pandas handles this correctly.")
    print(result)
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()