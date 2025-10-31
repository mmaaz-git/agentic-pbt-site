import pandas as pd
import tempfile
import os
from hypothesis import given, strategies as st, settings
from pandas.testing import assert_frame_equal

@settings(max_examples=100)
@given(
    data=st.lists(
        st.lists(st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text()), min_size=1, max_size=5),
        min_size=1,
        max_size=10
    ),
    sheet_name=st.one_of(st.just(0), st.just("Sheet1"))
)
def test_read_excel_excelfile_equivalence(data, sheet_name):
    if not data or not data[0]:
        return

    num_cols = len(data[0])
    if not all(len(row) == num_cols for row in data):
        return

    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df.to_excel(tmp_path, sheet_name="Sheet1", index=False)

        result_direct = pd.read_excel(tmp_path, sheet_name=sheet_name)

        excel_file = pd.ExcelFile(tmp_path)
        result_via_excelfile = excel_file.parse(sheet_name=sheet_name)
        excel_file.close()

        assert_frame_equal(result_direct, result_via_excelfile)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Test with the specific failing input
def test_failing_case():
    data = [[1.7976931348623155e+308]]

    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df.to_excel(tmp_path, sheet_name="Sheet1", index=False)

        result_direct = pd.read_excel(tmp_path, sheet_name="Sheet1")

        excel_file = pd.ExcelFile(tmp_path)
        result_via_excelfile = excel_file.parse(sheet_name="Sheet1")
        excel_file.close()

        assert_frame_equal(result_direct, result_via_excelfile)
        print("Test passed!")
    except OverflowError as e:
        print(f"OverflowError occurred: {e}")
        return False
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return True

if __name__ == "__main__":
    print("Running specific failing case...")
    test_failing_case()

    print("\nRunning hypothesis test...")
    test_read_excel_excelfile_equivalence()