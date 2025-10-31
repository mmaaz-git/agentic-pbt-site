import io
import pandas as pd
from hypothesis import given, strategies as st


@given(
    data=st.lists(
        st.lists(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=20), min_size=1, max_size=5),
        min_size=1,
        max_size=10
    )
)
def test_roundtrip_string_dataframe(data):
    num_cols = len(data[0])
    if not all(len(row) == num_cols for row in data):
        return

    for row in data:
        for val in row:
            if '\n' in val or '\r' in val:
                return

    col_names = [f"col{i}" for i in range(num_cols)]
    df = pd.DataFrame(data, columns=col_names)

    csv_str = df.to_csv(index=False)
    df_roundtrip = pd.read_csv(io.StringIO(csv_str))

    assert df.equals(df_roundtrip), f"Round-trip failed:\nOriginal:\n{df}\n\nRound-trip:\n{df_roundtrip}"

# Test with the specific failing input
if __name__ == "__main__":
    print("Testing with the specific failing input: data=[['']]")
    data = [['']]

    # Manually run the test logic without hypothesis
    num_cols = len(data[0])
    if not all(len(row) == num_cols for row in data):
        print("Skipping: rows have different lengths")
    else:
        for row in data:
            for val in row:
                if '\n' in val or '\r' in val:
                    print("Skipping: contains newlines")
                    exit()

        col_names = [f"col{i}" for i in range(num_cols)]
        df = pd.DataFrame(data, columns=col_names)

        csv_str = df.to_csv(index=False)
        print(f"CSV representation:\n{csv_str}")

        df_roundtrip = pd.read_csv(io.StringIO(csv_str))

        print(f"\nOriginal DataFrame:\n{df}")
        print(f"Original value type: {type(df.iloc[0, 0])}, repr: {repr(df.iloc[0, 0])}")
        print(f"\nRound-trip DataFrame:\n{df_roundtrip}")
        print(f"Round-trip value type: {type(df_roundtrip.iloc[0, 0])}, repr: {repr(df_roundtrip.iloc[0, 0])}")

        try:
            assert df.equals(df_roundtrip), f"Round-trip failed!"
            print("\n✓ Round-trip successful")
        except AssertionError as e:
            print(f"\n✗ {e}")
            print(f"df.equals(df_roundtrip) = {df.equals(df_roundtrip)}")