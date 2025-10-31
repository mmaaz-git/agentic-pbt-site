import io
import pandas as pd
from hypothesis import given, strategies as st, settings

# First, let's test the property-based test from the bug report
@given(
    text_data=st.lists(
        st.text(alphabet=st.characters(blacklist_categories=['Cs', 'Cc']), min_size=0, max_size=20),
        min_size=1,
        max_size=10
    ),
    num_cols=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=200)
def test_engine_equivalence_text(text_data, num_cols):
    columns = [f'col{i}' for i in range(num_cols)]
    data = {col: text_data for col in columns}
    df = pd.DataFrame(data)
    csv_str = df.to_csv(index=False)

    df_c = pd.read_csv(io.StringIO(csv_str), engine='c')
    df_python = pd.read_csv(io.StringIO(csv_str), engine='python')

    pd.testing.assert_frame_equal(df_c, df_python, check_dtype=True)

# Run the test to see if it fails
print("Running property-based test...")
try:
    test_engine_equivalence_text()
    print("Property-based test passed (no failures found)")
except Exception as e:
    print(f"Property-based test failed: {e}")

# Now test the specific failing case mentioned in the bug report
print("\nTesting specific case with empty string...")
text_data = ['']
num_cols = 1
columns = [f'col{i}' for i in range(num_cols)]
data = {col: text_data for col in columns}
df = pd.DataFrame(data)
csv_str = df.to_csv(index=False)
print(f"CSV string created from DataFrame: {repr(csv_str)}")

df_c = pd.read_csv(io.StringIO(csv_str), engine='c')
df_python = pd.read_csv(io.StringIO(csv_str), engine='python')

print(f"\nC engine result:")
print(df_c)
print(f"Shape: {df_c.shape}")
print(f"Values: {df_c.values}")

print(f"\nPython engine result:")
print(df_python)
print(f"Shape: {df_python.shape}")
print(f"Values: {df_python.values}")

try:
    pd.testing.assert_frame_equal(df_c, df_python, check_dtype=True)
    print("\nDataFrames are equal")
except AssertionError as e:
    print(f"\nDataFrames are NOT equal: {e}")