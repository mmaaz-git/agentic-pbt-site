#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, example
from pandas.io.sas.sasreader import read_sas
import pytest

@example("archive.xpt.backup")
@example("data.sas7bdat.old")
@example("my.xpt_notes.txt")
@given(st.text(min_size=1, max_size=50))
def test_format_detection_substring_bug(filename):
    if '.xpt' in filename.lower() or '.sas7bdat' in filename.lower():
        with pytest.raises(Exception):
            read_sas(filename)

# Run examples manually first
print("Testing specific examples:")
examples = ["archive.xpt.backup", "data.sas7bdat.old", "my.xpt_notes.txt"]

for example in examples:
    print(f"\nTesting: {example}")
    try:
        read_sas(example)
        print(f"  -> No exception raised (would have tried to read file)")
    except FileNotFoundError as e:
        print(f"  -> FileNotFoundError: File accepted as valid SAS format")
    except ValueError as e:
        print(f"  -> ValueError: {e}")
    except Exception as e:
        print(f"  -> {type(e).__name__}: {e}")

# Test some edge cases
print("\n\nAdditional edge cases:")
edge_cases = [
    "test.txt",  # No .xpt or .sas7bdat
    ".xpt",      # Just the extension
    ".sas7bdat", # Just the extension
    "file.XPT",  # uppercase
    "file.SAS7BDAT", # uppercase
]

for case in edge_cases:
    print(f"\nTesting: {case}")
    try:
        read_sas(case)
        print(f"  -> No exception raised (would have tried to read file)")
    except FileNotFoundError as e:
        print(f"  -> FileNotFoundError: File accepted as valid SAS format")
    except ValueError as e:
        print(f"  -> ValueError: {e}")
    except Exception as e:
        print(f"  -> {type(e).__name__}: {e}")