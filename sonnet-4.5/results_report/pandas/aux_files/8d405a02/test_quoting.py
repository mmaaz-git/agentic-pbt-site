import pandas as pd
import io
import csv

# Test with different quoting options
name = '\t'
df = pd.DataFrame([[42]], columns=[name])

print("Testing tab character column with different quoting options:")
print("=" * 60)

for quoting_option, quoting_name in [(csv.QUOTE_MINIMAL, "QUOTE_MINIMAL"),
                                      (csv.QUOTE_ALL, "QUOTE_ALL"),
                                      (csv.QUOTE_NONNUMERIC, "QUOTE_NONNUMERIC"),
                                      (csv.QUOTE_NONE, "QUOTE_NONE")]:
    try:
        csv_str = df.to_csv(index=False, quoting=quoting_option)
        print(f"\n{quoting_name}:")
        print(f"  CSV output: {repr(csv_str)}")

        result = pd.read_csv(io.StringIO(csv_str))
        print(f"  Result columns: {list(result.columns)}")
        print(f"  Result values: {result.values.tolist()}")
        print(f"  Data preserved: {result.columns[0] == name and result.values[0][0] == 42}")
    except Exception as e:
        print(f"\n{quoting_name}: ERROR - {e}")