from hypothesis import given, strategies as st, settings
from django.core.files.utils import validate_file_name
from django.core.exceptions import SuspiciousFileOperation
import pytest


@given(st.text(min_size=1), st.sampled_from(['/', '\\']))
@settings(max_examples=500)
def test_validate_file_name_rejects_path_separators(base_name, separator):
    name = f"{base_name}{separator}file"

    with pytest.raises(SuspiciousFileOperation):
        validate_file_name(name, allow_relative_path=False)

# Run the test
if __name__ == "__main__":
    test_validate_file_name_rejects_path_separators()