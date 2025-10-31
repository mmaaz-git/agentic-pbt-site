from hypothesis import given, strategies as st, settings, assume
import pandas as pd
from io import StringIO
import traceback

@settings(max_examples=10)  # Reduced for testing
@given(
    ncols=st.integers(min_value=2, max_value=10),
    nrows=st.integers(min_value=1, max_value=20),
    duplicate_index=st.integers(min_value=0, max_value=5)
)
def test_duplicate_index_col_no_crash(ncols, nrows, duplicate_index):
    assume(duplicate_index < ncols)

    data = [[i * ncols + j for j in range(ncols)] for i in range(nrows)]
    df = pd.DataFrame(data, columns=[f"col{i}" for i in range(ncols)])
    csv_data = df.to_csv(index=False)

    try:
        result = pd.read_csv(StringIO(csv_data), index_col=[duplicate_index, duplicate_index])
        print(f"Test passed: ncols={ncols}, nrows={nrows}, duplicate_index={duplicate_index}")
        return True
    except ValueError as e:
        if "list.remove(x): x not in list" in str(e):
            print(f"BUG REPRODUCED: ncols={ncols}, nrows={nrows}, duplicate_index={duplicate_index}")
            print(f"  Error: {e}")
            return False
        else:
            # Some other ValueError - unexpected
            raise

# Run the property-based test
print("Running property-based test...")
try:
    test_duplicate_index_col_no_crash()
    print("Property test completed")
except Exception as e:
    print(f"Property test failed with: {e}")
    traceback.print_exc()