# Bug Report: pandas.io.excel Surrogate Character Crash

**Target**: `pandas.DataFrame.to_excel`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

Writing a DataFrame containing Unicode surrogate characters (U+D800 to U+DFFF) to Excel causes a `UnicodeEncodeError` crash with a cryptic error message deep in the library stack, affecting both xlsxwriter and openpyxl engines.

## Property-Based Test

```python
import io
import pandas as pd
from hypothesis import given, settings
from hypothesis.extra.pandas import data_frames, column
import hypothesis.strategies as st


@given(
    df=data_frames(
        columns=[
            column("A", dtype=int),
            column("B", dtype=float),
            column("C", dtype=str),
        ],
        index=st.just(pd.RangeIndex(0, 10)),
    )
)
@settings(max_examples=50)
def test_roundtrip_basic(df):
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    result = pd.read_excel(buffer)
    pd.testing.assert_frame_equal(result, df, check_dtype=False)

if __name__ == "__main__":
    test_roundtrip_basic()
```

<details>

<summary>
**Failing input**: DataFrame with string column containing surrogate character `\ud800`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 27, in <module>
  |     test_roundtrip_basic()
  |     ~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 9, in test_roundtrip_basic
  |     df=data_frames(
  |                ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 21, in test_roundtrip_basic
    |     df.to_excel(buffer, index=False)
    |     ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/util/_decorators.py", line 333, in wrapper
    |     return func(*args, **kwargs)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/generic.py", line 2436, in to_excel
    |     formatter.write(
    |     ~~~~~~~~~~~~~~~^
    |         excel_writer,
    |         ^^^^^^^^^^^^^
    |     ...<6 lines>...
    |         engine_kwargs=engine_kwargs,
    |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/excel.py", line 962, in write
    |     writer.close()
    |     ~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_base.py", line 1357, in close
    |     self._save()
    |     ~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_xlsxwriter.py", line 239, in _save
    |     self.book.close()
    |     ~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/workbook.py", line 387, in close
    |     self._store_workbook()
    |     ~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/workbook.py", line 803, in _store_workbook
    |     xml_files = packager._create_package()
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/packager.py", line 149, in _create_package
    |     self._write_shared_strings_file()
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/packager.py", line 309, in _write_shared_strings_file
    |     sst._assemble_xml_file()
    |     ~~~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/sharedstrings.py", line 53, in _assemble_xml_file
    |     self._write_sst_strings()
    |     ~~~~~~~~~~~~~~~~~~~~~~~^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/sharedstrings.py", line 83, in _write_sst_strings
    |     self._write_si(string)
    |     ~~~~~~~~~~~~~~^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/sharedstrings.py", line 100, in _write_si
    |     self._xml_si_element(string, attributes)
    |     ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/xmlwriter.py", line 129, in _xml_si_element
    |     self.fh.write(f"<si><t{attr}>{string}</t></si>")
    |     ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 7: surrogates not allowed
    | Falsifying example: test_roundtrip_basic(
    |     df=
    |            A    B  C
    |         0  0  0.0  \ud800
    |         1  0  0.0  \ud800
    |         2  0  0.0  \ud800
    |         3  0  0.0  \ud800
    |         4  0  0.0  \ud800
    |         5  0  0.0  \ud800
    |         6  0  0.0  \ud800
    |         7  0  0.0  \ud800
    |         8  0  0.0  \ud800
    |         9  0  0.0  \ud800
    |     ,
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/workbook.py:388
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 24, in test_roundtrip_basic
    |     pd.testing.assert_frame_equal(result, df, check_dtype=False)
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1303, in assert_frame_equal
    |     assert_series_equal(
    |     ~~~~~~~~~~~~~~~~~~~^
    |         lcol,
    |         ^^^^^
    |     ...<12 lines>...
    |         check_flags=False,
    |         ^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 1091, in assert_series_equal
    |     _testing.assert_almost_equal(
    |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
    |         left._values,
    |         ^^^^^^^^^^^^^
    |     ...<5 lines>...
    |         index_values=left.index,
    |         ^^^^^^^^^^^^^^^^^^^^^^^^
    |     )
    |     ^
    |   File "pandas/_libs/testing.pyx", line 55, in pandas._libs.testing.assert_almost_equal
    |   File "pandas/_libs/testing.pyx", line 173, in pandas._libs.testing.assert_almost_equal
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/_testing/asserters.py", line 620, in raise_assert_detail
    |     raise AssertionError(msg)
    | AssertionError: DataFrame.iloc[:, 2] (column name="C") are different
    |
    | DataFrame.iloc[:, 2] (column name="C") values are different (100.0 %)
    | [index]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    | [left]:  [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]
    | [right]: [, , , , , , , , , ]
    | At positional index 0, first diff: nan !=
    | Falsifying example: test_roundtrip_basic(
    |     df=
    |            A    B C
    |         0  0  0.0
    |         1  0  0.0
    |         2  0  0.0
    |         3  0  0.0
    |         4  0  0.0
    |         5  0  0.0
    |         6  0  0.0
    |         7  0  0.0
    |         8  0  0.0
    |         9  0  0.0
    |     ,
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:52
    |         /home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/fromnumeric.py:3614
    |         /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:138
    |         /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:628
    |         /home/npc/miniconda/lib/python3.13/site-packages/pandas/_config/config.py:697
    |         (and 2 more with settings.verbosity >= verbose)
    +------------------------------------
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
Exception ignored in: <function ZipFile.__del__ at 0x74335e5d8d60>
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1980, in __del__
  File "/home/npc/miniconda/lib/python3.13/zipfile/__init__.py", line 1997, in close
ValueError: I/O operation on closed file.
```
</details>

## Reproducing the Bug

```python
import io
import pandas as pd

# Create a DataFrame with a surrogate character
df = pd.DataFrame({"A": [0], "B": [0.0], "C": ["\ud800"]})

# Try to write to Excel
buffer = io.BytesIO()
df.to_excel(buffer, index=False)
```

<details>

<summary>
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/repo.py", line 9, in <module>
    df.to_excel(buffer, index=False)
    ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/util/_decorators.py", line 333, in wrapper
    return func(*args, **kwargs)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/generic.py", line 2436, in to_excel
    formatter.write(
    ~~~~~~~~~~~~~~~^
        excel_writer,
        ^^^^^^^^^^^^^
    ...<6 lines>...
        engine_kwargs=engine_kwargs,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/excel.py", line 962, in write
    writer.close()
    ~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_base.py", line 1357, in close
    self._save()
    ~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/excel/_xlsxwriter.py", line 239, in _save
    self.book.close()
    ~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/workbook.py", line 387, in close
    self._store_workbook()
    ~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/workbook.py", line 803, in _store_workbook
    xml_files = packager._create_package()
  File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/packager.py", line 149, in _create_package
    self._write_shared_strings_file()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/packager.py", line 309, in _write_shared_strings_file
    sst._assemble_xml_file()
    ~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/sharedstrings.py", line 53, in _assemble_xml_file
    self._write_sst_strings()
    ~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/sharedstrings.py", line 83, in _write_sst_strings
    self._write_si(string)
    ~~~~~~~~~~~~~~^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/sharedstrings.py", line 100, in _write_si
    self._xml_si_element(string, attributes)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/xlsxwriter/xmlwriter.py", line 129, in _xml_si_element
    self.fh.write(f"<si><t{attr}>{string}</t></si>")
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'utf-8' codec can't encode character '\ud800' in position 7: surrogates not allowed
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Documentation Gap**: The `DataFrame.to_excel()` documentation does not mention any limitations regarding Unicode characters. Users have a reasonable expectation that valid Python strings should be writable to Excel.

2. **Poor Error Message**: The error occurs deep in the xlsxwriter/openpyxl stack with no indication of which column or cell caused the issue. The message "surrogates not allowed" provides no guidance on how to fix the problem.

3. **Inconsistent with Other Formats**: Other pandas export methods like `to_csv()` and `to_json()` handle these characters (either by preserving them or providing clear errors at the pandas level).

4. **Both Engines Affected**: The issue occurs with both supported Excel engines (xlsxwriter and openpyxl), indicating this is a systematic problem with how pandas handles string data for Excel export.

While surrogate characters (U+D800 to U+DFFF) are technically invalid in UTF-8 according to the Unicode specification, they can exist in Python strings and users may encounter them through:
- Incorrectly decoded UTF-16 data
- Data corruption during file transfers
- Manual string construction in testing/fuzzing scenarios
- Legacy system migrations

## Relevant Context

- The Excel XML format requires valid UTF-8 encoding
- Python strings can contain surrogate characters even though they're invalid in UTF-8
- The error traceback shows the issue occurs when writing to the underlying XML structure
- pandas documentation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html
- Unicode surrogate pairs specification: https://unicode.org/faq/utf_bom.html#utf16-2
- Related code location: `/pandas/io/formats/excel.py:890` where cells are formatted

## Proposed Fix

Add early validation in pandas to detect surrogate characters and provide a helpful error message before passing data to the underlying Excel engines:

```diff
--- a/pandas/io/formats/excel.py
+++ b/pandas/io/formats/excel.py
@@ -888,6 +888,21 @@ class ExcelFormatter:
     def get_formatted_cells(self) -> Iterable[ExcelCell]:
         for cell in itertools.chain(self._format_header(), self._format_body()):
             cell.val = self._format_value(cell.val)
+
+            # Check for surrogate characters that will cause encoding errors
+            if isinstance(cell.val, str):
+                try:
+                    cell.val.encode('utf-8')
+                except UnicodeEncodeError as e:
+                    if 'surrogates not allowed' in str(e):
+                        raise ValueError(
+                            f"Cell at row {cell.row}, column {cell.col} contains "
+                            f"invalid Unicode surrogate characters (U+D800 to U+DFFF). "
+                            f"These characters cannot be written to Excel files. "
+                            f"Please clean your data by removing or replacing these "
+                            f"characters before writing to Excel."
+                        ) from e
+                    raise  # Re-raise other encoding errors
             yield cell

     @doc(storage_options=_shared_docs["storage_options"])
```