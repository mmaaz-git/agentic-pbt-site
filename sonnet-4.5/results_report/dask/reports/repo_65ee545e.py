from dask.dataframe.io.orc.arrow import ArrowORCEngine
from unittest.mock import MagicMock
import pyarrow as pa
import pyarrow.orc as orc

# Create mock filesystem and file objects
mock_fs = MagicMock()
paths = ['file1.orc', 'file2.orc', 'file3.orc']

# Mock file object
mock_file = MagicMock()
mock_fs.open.return_value.__enter__.return_value = mock_file

# Mock ORC file with a schema
mock_orc_file = MagicMock()
mock_orc_file.schema = pa.schema([('col1', pa.int64())])

# Store the original ORCFile to restore it later
original_orcfile = orc.ORCFile

# Replace ORCFile with our mock
orc.ORCFile = lambda f: mock_orc_file

try:
    # Call the read_metadata method with split_stripes=False
    parts, schema, meta = ArrowORCEngine.read_metadata(
        mock_fs, paths, None, None, False, None
    )

    # Extract the files that were opened
    opened_files = [call[0][0] for call in mock_fs.open.call_args_list]

    print("Files that should be processed:", paths)
    print("Files actually opened:", opened_files)
    print()

    # Check if the bug exists
    if len(opened_files) == 1 and opened_files[0] == 'file1.orc':
        print("BUG CONFIRMED: Only paths[0] (file1.orc) was opened!")
        print("Expected: Each file should be checked when processing it")
        print("Actual: Only the first file is ever opened to read schema")
    else:
        print("Bug not reproduced - all files were opened correctly")

finally:
    # Restore the original ORCFile
    orc.ORCFile = original_orcfile