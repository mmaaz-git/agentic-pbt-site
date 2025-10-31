import tempfile
import shutil
import pandas as pd
import dask.dataframe as dd
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes


# First, let's test the basic reproduction case
def test_basic_reproduction():
    """Test the basic bug reproduction from the report"""
    tmpdir = tempfile.mkdtemp()
    try:
        pdf = pd.DataFrame({
            'int_col': [0, 1, 2, 3],
            'float_col': [0.0, 1.0, 2.0, 3.0],
        })

        print("=== Basic Reproduction Test ===")
        print(f"Original index: {list(pdf.index)}")

        ddf = dd.from_pandas(pdf, npartitions=2)
        orc_path = f"{tmpdir}/test_orc"
        ddf.to_orc(orc_path, write_index=True)

        result_ddf = dd.read_orc(orc_path)
        result_pdf = result_ddf.compute()

        print(f"Result index:   {list(result_pdf.index)}")

        if list(pdf.index) == list(result_pdf.index):
            print("✓ Index preserved correctly")
        else:
            print(f"✗ Index mismatch: {list(pdf.index)} != {list(result_pdf.index)}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_custom_index():
    """Test with custom index values"""
    tmpdir = tempfile.mkdtemp()
    try:
        pdf = pd.DataFrame({
            'int_col': [10, 20, 30, 40],
        }, index=[100, 200, 300, 400])

        print("\n=== Custom Index Test ===")
        print(f"Original index: {list(pdf.index)}")

        ddf = dd.from_pandas(pdf, npartitions=2)
        orc_path = f"{tmpdir}/test_orc"
        ddf.to_orc(orc_path, write_index=True)

        result_ddf = dd.read_orc(orc_path)
        result = result_ddf.compute()

        print(f"Result index:   {list(result.index)}")

        if list(pdf.index) == list(result.index):
            print("✓ Custom index preserved correctly")
        else:
            print(f"✗ Index mismatch: {list(pdf.index)} != {list(result.index)}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_single_partition():
    """Test with single partition to see if it works correctly"""
    tmpdir = tempfile.mkdtemp()
    try:
        pdf = pd.DataFrame({
            'int_col': [0, 1, 2, 3],
            'float_col': [0.0, 1.0, 2.0, 3.0],
        })

        print("\n=== Single Partition Test ===")
        print(f"Original index: {list(pdf.index)}")

        ddf = dd.from_pandas(pdf, npartitions=1)  # Single partition
        orc_path = f"{tmpdir}/test_orc_single"
        ddf.to_orc(orc_path, write_index=True)

        result_ddf = dd.read_orc(orc_path)
        result_pdf = result_ddf.compute()

        print(f"Result index:   {list(result_pdf.index)}")

        if list(pdf.index) == list(result_pdf.index):
            print("✓ Index preserved correctly with single partition")
        else:
            print(f"✗ Index mismatch: {list(pdf.index)} != {list(result_pdf.index)}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_minimal_failing_case():
    """Test the minimal failing case from the report"""
    tmpdir = tempfile.mkdtemp()
    try:
        pdf = pd.DataFrame({
            'int_col': [0, 0],
            'float_col': [0.0, 0.0],
            'str_col': ['', '']
        })

        print("\n=== Minimal Failing Case ===")
        print(f"Original DataFrame:\n{pdf}")
        print(f"Original index: {list(pdf.index)}")

        ddf = dd.from_pandas(pdf, npartitions=2)
        orc_path = f"{tmpdir}/test_orc"
        ddf.to_orc(orc_path, write_index=True)

        result_ddf = dd.read_orc(orc_path)
        result_pdf = result_ddf.compute()

        print(f"Result DataFrame:\n{result_pdf}")
        print(f"Result index:   {list(result_pdf.index)}")

        # Check both index and data
        try:
            pd.testing.assert_frame_equal(pdf, result_pdf, check_dtype=False)
            print("✓ DataFrame preserved correctly")
        except AssertionError as e:
            print(f"✗ DataFrame mismatch: {e}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_hypothesis_property():
    """Run the hypothesis property test"""
    print("\n=== Running Hypothesis Property Test ===")
    failures = []

    @given(
        data_frames(
            columns=[
                column("int_col", dtype=int, elements=st.integers(min_value=-1000, max_value=1000)),
                column("float_col", dtype=float, elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)),
                column("str_col", dtype=str, elements=st.text(min_size=0, max_size=20)),
            ],
            index=range_indexes(min_size=1, max_size=100),
        )
    )
    @settings(max_examples=10, deadline=None)
    def test_orc_round_trip_preserves_data(pdf):
        tmpdir = tempfile.mkdtemp()
        try:
            ddf = dd.from_pandas(pdf, npartitions=min(2, len(pdf)))  # At least 2 partitions if possible
            orc_path = f"{tmpdir}/test_orc"
            ddf.to_orc(orc_path, write_index=True)
            result_ddf = dd.read_orc(orc_path)
            result_pdf = result_ddf.compute()

            try:
                pd.testing.assert_frame_equal(pdf, result_pdf, check_dtype=False)
            except AssertionError as e:
                failures.append({
                    'original_index': list(pdf.index),
                    'result_index': list(result_pdf.index),
                    'error': str(e)
                })

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    try:
        test_orc_round_trip_preserves_data()
        print("Hypothesis tests completed")
    except Exception as e:
        print(f"Hypothesis test error: {e}")

    if failures:
        print(f"Found {len(failures)} failures in hypothesis testing")
        for i, fail in enumerate(failures[:3]):  # Show first 3 failures
            print(f"  Failure {i+1}:")
            print(f"    Original index: {fail['original_index'][:10]}...")
            print(f"    Result index: {fail['result_index'][:10]}...")


if __name__ == "__main__":
    test_basic_reproduction()
    test_custom_index()
    test_single_partition()
    test_minimal_failing_case()
    test_hypothesis_property()