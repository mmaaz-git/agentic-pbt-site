from hypothesis import given, strategies as st, settings
from dask.dataframe.io.orc.arrow import ArrowORCEngine
from unittest.mock import MagicMock
import pyarrow as pa
import pyarrow.orc as orc

@given(st.lists(st.text(min_size=1, alphabet=st.characters(blacklist_characters=['/', '\x00', '\\'])),
                min_size=2, max_size=5, unique=True))
@settings(max_examples=50)
def test_read_metadata_opens_correct_files(file_names):
    paths = [f"{name}.orc" for name in file_names]

    mock_fs = MagicMock()
    mock_file = MagicMock()
    mock_fs.open.return_value.__enter__.return_value = mock_file

    mock_orc_file = MagicMock()
    mock_orc_file.schema = pa.schema([('col1', pa.int64())])

    # Store original to restore later
    original_orcfile = orc.ORCFile

    # Mock the ORCFile
    orc.ORCFile = lambda f: mock_orc_file

    try:
        ArrowORCEngine.read_metadata(
            mock_fs, paths, None, None, False, None
        )

        opened_paths = [call_args[0][0] for call_args in mock_fs.open.call_args_list]

        # When split_stripes=False, the bug causes only paths[0] to be opened
        if len(paths) > 1:
            assert paths[0] in opened_paths, f"First file {paths[0]} should be opened"
            if len(set(opened_paths)) == 1 and opened_paths[0] == paths[0]:
                # Bug detected: Only first file was opened multiple times or once
                print(f"BUG: With {len(paths)} files, only {paths[0]} was opened")
                print(f"Expected to open at minimum: {paths[0]} (for schema)")
                print(f"Actually opened: {opened_paths}")
                assert False, f"Only {paths[0]} was opened, but should open path in loop, not paths[0]"
    finally:
        # Restore original
        orc.ORCFile = original_orcfile

# Run the test
if __name__ == "__main__":
    test_read_metadata_opens_correct_files()