import pandas as pd
from io import StringIO
from hypothesis import given, strategies as st, settings
import pytest
import re

print("Testing pandas.read_csv with regex special characters in separators")
print("="*70)

# First, run the hypothesis test
regex_special_chars = ['|', '+', '*', '?', '.', '^', '$']

@given(
    special_char=st.sampled_from(regex_special_chars),
    num_cols=st.integers(min_value=2, max_value=5),
    num_rows=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=100)
def test_regex_special_char_separators(special_char, num_cols, num_rows):
    separator = special_char * 2

    header = separator.join([f'col{i}' for i in range(num_cols)])
    rows = []
    for i in range(num_rows):
        row = separator.join([str(i * num_cols + j) for j in range(num_cols)])
        rows.append(row)

    csv_content = header + '\n' + '\n'.join(rows)

    try:
        df = pd.read_csv(StringIO(csv_content), sep=separator, engine='python')

        if df.shape[1] != num_cols:
            expected_cols = [f'col{i}' for i in range(num_cols)]
            actual_cols = list(df.columns)
            print(f"FAILURE: Separator {repr(separator)} failed: expected {num_cols} columns {expected_cols}, "
                  f"got {df.shape[1]} columns {actual_cols}")
    except (pd.errors.ParserError, ValueError, re.error) as e:
        print(f"EXCEPTION: Separator {repr(separator)} caused error: {type(e).__name__}: {e}")

print("\nRunning hypothesis test...")
test_regex_special_char_separators()

print("\n" + "="*70)
print("Testing specific cases manually:")
print("="*70)

# Case 1: || separator (wrong column count)
print("\nCase 1: '||' separator")
try:
    csv_data = 'col0||col1\n0||1'
    df = pd.read_csv(StringIO(csv_data), sep='||', engine='python')

    print(f"Input CSV: {repr(csv_data)}")
    print(f"Expected: 2 columns ['col0', 'col1']")
    print(f"Got: {df.shape[1]} columns {list(df.columns)}")
    print(f"DataFrame:\n{df}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Case 2: .. separator (wrong column count)
print("\nCase 2: '..' separator")
try:
    csv_data = 'col0..col1\n0..1'
    df = pd.read_csv(StringIO(csv_data), sep='..', engine='python')

    print(f"Input CSV: {repr(csv_data)}")
    print(f"Expected: 2 columns ['col0', 'col1']")
    print(f"Got: {df.shape[1]} columns {list(df.columns)}")
    print(f"DataFrame:\n{df}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Case 3: ++ separator (regex error)
print("\nCase 3: '++' separator")
try:
    csv_data = 'col0++col1\n0++1'
    df = pd.read_csv(StringIO(csv_data), sep='++', engine='python')

    print(f"Input CSV: {repr(csv_data)}")
    print(f"Expected: 2 columns ['col0', 'col1']")
    print(f"Got: {df.shape[1]} columns {list(df.columns)}")
    print(f"DataFrame:\n{df}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Case 4: ** separator
print("\nCase 4: '**' separator")
try:
    csv_data = 'col0**col1\n0**1'
    df = pd.read_csv(StringIO(csv_data), sep='**', engine='python')

    print(f"Input CSV: {repr(csv_data)}")
    print(f"Expected: 2 columns ['col0', 'col1']")
    print(f"Got: {df.shape[1]} columns {list(df.columns)}")
    print(f"DataFrame:\n{df}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Case 5: ?? separator
print("\nCase 5: '??' separator")
try:
    csv_data = 'col0??col1\n0??1'
    df = pd.read_csv(StringIO(csv_data), sep='??', engine='python')

    print(f"Input CSV: {repr(csv_data)}")
    print(f"Expected: 2 columns ['col0', 'col1']")
    print(f"Got: {df.shape[1]} columns {list(df.columns)}")
    print(f"DataFrame:\n{df}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*70)
print("Testing workarounds:")
print("="*70)

# Test workaround with re.escape
print("\nWorkaround 1: Using re.escape('||')")
try:
    csv_data = 'col0||col1\n0||1'
    df = pd.read_csv(StringIO(csv_data), sep=re.escape('||'), engine='python')

    print(f"Input CSV: {repr(csv_data)}")
    print(f"Separator used: {repr(re.escape('||'))}")
    print(f"Got: {df.shape[1]} columns {list(df.columns)}")
    print(f"DataFrame:\n{df}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test workaround with explicit escaping
print("\nWorkaround 2: Using explicit escaping r'\\|\\|'")
try:
    csv_data = 'col0||col1\n0||1'
    df = pd.read_csv(StringIO(csv_data), sep=r'\|\|', engine='python')

    print(f"Input CSV: {repr(csv_data)}")
    print(f"Separator used: r'\\|\\|'")
    print(f"Got: {df.shape[1]} columns {list(df.columns)}")
    print(f"DataFrame:\n{df}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")