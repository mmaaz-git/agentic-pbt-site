# Bug Report: pydantic.experimental.pipeline datetime_tz() Method Not Implemented

**Target**: `pydantic.experimental.pipeline._Pipeline.datetime_tz`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `datetime_tz()` method in pydantic's experimental pipeline module is exposed as a public API method but always raises `NotImplementedError` when called with any timezone, violating the API contract and making the method completely unusable.

## Property-Based Test

```python
import datetime
from typing import Annotated
from hypothesis import given, strategies as st
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as

@given(st.datetimes(timezones=st.timezones()))
def test_datetime_tz_should_work(dt):
    pipeline = validate_as(datetime.datetime).datetime_tz(dt.tzinfo)

    class TestModel(BaseModel):
        field: Annotated[datetime.datetime, pipeline]

    model = TestModel(field=dt)
    assert model.field.tzinfo == dt.tzinfo

test_datetime_tz_should_work()
```

<details>

<summary>
**Failing input**: `datetime.datetime(2000, 1, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC'))`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 17, in <module>
    test_datetime_tz_should_work()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 8, in test_datetime_tz_should_work
    def test_datetime_tz_should_work(dt):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/14/hypo.py", line 11, in test_datetime_tz_should_work
    class TestModel(BaseModel):
        field: Annotated[datetime.datetime, pipeline]
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_model_construction.py", line 226, in __new__
    complete_model_class(
    ~~~~~~~~~~~~~~~~~~~~^
        cls,
        ^^^^
    ...<4 lines>...
        create_model_module=_create_model_module,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_model_construction.py", line 658, in complete_model_class
    schema = cls.__get_pydantic_core_schema__(cls, handler)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/main.py", line 702, in __get_pydantic_core_schema__
    return handler(source)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_schema_generation_shared.py", line 84, in __call__
    schema = self._handler(source_type)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py", line 612, in generate_schema
    schema = self._generate_schema_inner(obj)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py", line 881, in _generate_schema_inner
    return self._model_schema(obj)
           ~~~~~~~~~~~~~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py", line 693, in _model_schema
    {k: self._generate_md_field_schema(k, v, decorators) for k, v in fields.items()},
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py", line 1073, in _generate_md_field_schema
    common_field = self._common_field_schema(name, field_info, decorators)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py", line 1265, in _common_field_schema
    schema = self._apply_annotations(
        source_type,
        annotations + validators_from_decorators,
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py", line 2062, in _apply_annotations
    schema = get_inner_schema(source_type)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_schema_generation_shared.py", line 84, in __call__
    schema = self._handler(source_type)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py", line 2137, in new_handler
    schema = metadata_get_schema(source, get_inner_schema)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/pipeline.py", line 347, in __get_pydantic_core_schema__
    s = _apply_step(step, s, handler, source_type)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/pipeline.py", line 387, in _apply_step
    s = _apply_constraint(s, step.constraint)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/pipeline.py", line 575, in _apply_constraint
    raise NotImplementedError('Constraining to a specific timezone is not yet supported')
NotImplementedError: Constraining to a specific timezone is not yet supported
Falsifying example: test_datetime_tz_should_work(
    dt=datetime.datetime(2000, 1, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC')),  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import datetime
from typing import Annotated
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as

pipeline = validate_as(datetime.datetime).datetime_tz(datetime.timezone.utc)

class TestModel(BaseModel):
    field: Annotated[datetime.datetime, pipeline]

dt = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
model = TestModel(field=dt)
```

<details>

<summary>
NotImplementedError: Constraining to a specific timezone is not yet supported
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/14/repo.py", line 8, in <module>
    class TestModel(BaseModel):
        field: Annotated[datetime.datetime, pipeline]
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_model_construction.py", line 226, in __new__
    complete_model_class(
    ~~~~~~~~~~~~~~~~~~~~^
        cls,
        ^^^^
    ...<4 lines>...
        create_model_module=_create_model_module,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_model_construction.py", line 658, in complete_model_class
    schema = cls.__get_pydantic_core_schema__(cls, handler)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/main.py", line 702, in __get_pydantic_core_schema__
    return handler(source)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_schema_generation_shared.py", line 84, in __call__
    schema = self._handler(source_type)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py", line 612, in generate_schema
    schema = self._generate_schema_inner(obj)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py", line 881, in _generate_schema_inner
    return self._model_schema(obj)
           ~~~~~~~~~~~~~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py", line 693, in _model_schema
    {k: self._generate_md_field_schema(k, v, decorators) for k, v in fields.items()},
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py", line 1073, in _generate_md_field_schema
    common_field = self._common_field_schema(name, field_info, decorators)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py", line 1265, in _common_field_schema
    schema = self._apply_annotations(
        source_type,
        annotations + validators_from_decorators,
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py", line 2062, in _apply_annotations
    schema = get_inner_schema(source_type)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_schema_generation_shared.py", line 84, in __call__
    schema = self._handler(source_type)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/_internal/_generate_schema.py", line 2137, in new_handler
    schema = metadata_get_schema(source, get_inner_schema)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/pipeline.py", line 347, in __get_pydantic_core_schema__
    s = _apply_step(step, s, handler, source_type)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/pipeline.py", line 387, in _apply_step
    s = _apply_constraint(s, step.constraint)
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/pipeline.py", line 575, in _apply_constraint
    raise NotImplementedError('Constraining to a specific timezone is not yet supported')
NotImplementedError: Constraining to a specific timezone is not yet supported
```
</details>

## Why This Is A Bug

This is a clear **contract violation** in the API design. The `datetime_tz()` method is publicly exposed in the pipeline API (lines 290-293 of pipeline.py) with proper type hints and no indication that it doesn't work:

```python
def datetime_tz(
    self: _Pipeline[_InT, datetime.datetime], tz: datetime.tzinfo
) -> _Pipeline[_InT, datetime.datetime]:
    return self.constrain(annotated_types.Timezone(tz))  # type: ignore
```

However, when this method is called with ANY timezone (not just specific ones), it creates an `annotated_types.Timezone` constraint that the constraint handler explicitly rejects at lines 574-575:

```python
else:
    raise NotImplementedError('Constraining to a specific timezone is not yet supported')
```

This means the method is **completely non-functional** - it will always fail regardless of input. Users have no way to know this without trying to use it and encountering the error.

The issue is particularly misleading because:

1. **Similar methods work correctly**: `datetime_tz_naive()` and `datetime_tz_aware()` both function properly, creating a reasonable expectation that `datetime_tz()` should also work.

2. **Type hints suggest it works**: The method has complete type annotations indicating it returns a valid `_Pipeline` object.

3. **No documentation of limitation**: There's no docstring, warning, or other indication that this method is not implemented.

4. **The `# type: ignore` comment** on line 293 suggests developers were aware of some issue but chose to suppress it rather than address the fundamental problem.

## Relevant Context

The experimental pipeline module provides a fluent API for building validation and transformation chains in Pydantic. The module includes several datetime-related methods:

- `datetime_tz_naive()` - Validates datetime has no timezone (works)
- `datetime_tz_aware()` - Validates datetime has any timezone (works)
- `datetime_tz(tz)` - Should validate datetime has specific timezone (broken)
- `datetime_with_tz(tz)` - Transforms datetime to have specific timezone (works)

The distinction between constraining (validation) and transforming is important: `datetime_tz()` should validate that an incoming datetime already has the specified timezone, while `datetime_with_tz()` modifies the datetime to have a different timezone. These are fundamentally different operations serving different use cases.

The experimental module documentation at `/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py` warns that the module is "subject to change and deprecation," but this doesn't excuse having completely non-functional methods in the public API.

## Proposed Fix

The most appropriate fix is to implement the missing functionality in the constraint handler:

```diff
@@ -572,7 +572,11 @@ def _apply_constraint(

             s = _check_func(check_tz_naive, 'timezone naive', s)
     else:
-        raise NotImplementedError('Constraining to a specific timezone is not yet supported')
+        def check_specific_tz(v: object) -> bool:
+            assert isinstance(v, datetime.datetime)
+            return v.tzinfo is not None and v.tzinfo.tzname(v) == tz.tzname(v)
+
+        s = _check_func(check_specific_tz, f'timezone {tz.tzname(None) if hasattr(tz, "tzname") else tz}', s)
 elif isinstance(constraint, annotated_types.Interval):
     if constraint.ge:
         s = _apply_constraint(s, annotated_types.Ge(constraint.ge))
```

Alternative approaches if full implementation is not feasible:
1. Remove the method from public API until ready
2. Raise NotImplementedError immediately in the method body with clear documentation
3. Add a warning in the docstring that the method is not yet implemented