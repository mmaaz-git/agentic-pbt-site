# Bug Report: Django Field Validation Incorrectly Rejects Empty Values in Choices

**Target**: `django.db.models.fields.Field.validate`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Django field validation incorrectly rejects empty values (empty string, None) even when they are explicitly included as valid choices. The validation logic skips choice validation for empty values then rejects them in blank/null validation, preventing legitimate use of empty values as choices.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import django
from django.conf import settings as django_settings
django_settings.configure(USE_I18N=True, USE_TZ=False)
django.setup()

from django.db.models.fields import CharField
from django.core.exceptions import ValidationError
import pytest


@given(
    st.lists(st.tuples(st.text(), st.text()), min_size=1),
    st.text()
)
@settings(max_examples=100)
def test_charfield_validates_choices(choices, value):
    """
    Property: CharField should accept values in choices, reject others.
    """
    field = CharField(choices=choices)
    choice_values = {choice[0] for choice in choices}

    if value in choice_values:
        field.validate(value, None)
    else:
        with pytest.raises(ValidationError):
            field.validate(value, None)
```

<details>

<summary>
**Failing input**: `choices=[('', '')], value=''`
</summary>
```
Running property-based test for CharField validation...
Testing with failing input: choices=[('', '')], value=''

choices = [('', '')]
value = ''
value in choice_values = True

FAIL: Validation failed with error: ['This field cannot be blank.']
This is a BUG: empty string is in choices but was rejected!

============================================================
Running full Hypothesis test suite...

Property test failed!
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 62, in <module>
    test_charfield_validates_choices()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 13, in test_charfield_validates_choices
    st.lists(st.tuples(st.text(), st.text()), min_size=1),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 25, in test_charfield_validates_choices
    field.validate(value, None)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/db/models/fields/__init__.py", line 827, in validate
    raise exceptions.ValidationError(self.error_messages["blank"], code="blank")
django.core.exceptions.ValidationError: ['This field cannot be blank.']
Falsifying example: test_charfield_validates_choices(
    choices=[('', '')],
    value='',
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/58/hypo.py:25
```
</details>

## Reproducing the Bug

```python
import django
from django.conf import settings
settings.configure(USE_I18N=True, USE_TZ=False)
django.setup()

from django.db.models.fields import CharField, IntegerField
from django.core.exceptions import ValidationError

print("Testing CharField with empty string in choices (blank=False)...")
field = CharField(choices=[('', 'Empty choice'), ('a', 'A choice')], blank=False)

try:
    field.validate('', None)
    print("SUCCESS: Empty string validated successfully")
except ValidationError as e:
    print(f"ERROR: {e}")

print("\nTesting IntegerField with None in choices (null=False)...")
field2 = IntegerField(choices=[(None, 'None choice'), (1, 'One')], null=False)

try:
    field2.validate(None, None)
    print("SUCCESS: None validated successfully")
except ValidationError as e:
    print(f"ERROR: {e}")

print("\nControl test: Non-empty value 'a' in choices...")
try:
    field.validate('a', None)
    print("SUCCESS: 'a' validated successfully")
except ValidationError as e:
    print(f"ERROR: {e}")

print("\nControl test: Value 'b' not in choices...")
try:
    field.validate('b', None)
    print("SUCCESS: 'b' validated successfully")
except ValidationError as e:
    print(f"ERROR: {e}")

print("\nControl test: Empty string with blank=True...")
field3 = CharField(choices=[('', 'Empty'), ('a', 'A')], blank=True)
try:
    field3.validate('', None)
    print("SUCCESS: Empty string with blank=True validated successfully")
except ValidationError as e:
    print(f"ERROR: {e}")
```

<details>

<summary>
ValidationError raised when empty values are explicitly in choices
</summary>
```
Testing CharField with empty string in choices (blank=False)...
ERROR: ['This field cannot be blank.']

Testing IntegerField with None in choices (null=False)...
ERROR: ['This field cannot be null.']

Control test: Non-empty value 'a' in choices...
SUCCESS: 'a' validated successfully

Control test: Value 'b' not in choices...
ERROR: ["Value 'b' is not a valid choice."]

Control test: Empty string with blank=True...
SUCCESS: Empty string with blank=True validated successfully
```
</details>

## Why This Is A Bug

The bug stems from flawed validation logic in `django/db/models/fields/__init__.py` at line 807:

```python
if self.choices is not None and value not in self.empty_values:
    # Check if value is in choices
    ...
```

The condition `value not in self.empty_values` causes choice validation to be skipped entirely when the value is empty (empty string, None, [], (), {}). This creates the following problematic flow:

1. **Empty value arrives**: `value = ''` or `value = None`
2. **Choice validation skipped**: Because `value in self.empty_values`, the condition evaluates to False and choice validation is bypassed
3. **Blank/null validation runs**: Lines 823-827 check blank/null constraints and reject the empty value
4. **Result**: Empty values are rejected even when explicitly included in the choices list

This violates the principle of specificity - when a developer explicitly includes an empty value in the choices list, they are declaring it as a valid option. The current implementation makes it impossible to have empty values as valid choices without also setting `blank=True` or `null=True`, which may not be desired for data integrity reasons.

The Django documentation does not specify this behavior, and it contradicts reasonable developer expectations. When `choices=[('', 'Empty')]` is specified, developers expect the empty string to be a valid choice regardless of the `blank` setting.

## Relevant Context

Django's `empty_values` is defined as `(None, '', [], (), {})` and is used throughout the framework to identify "empty" inputs. However, the interaction between choice validation and blank/null validation creates an unexpected precedence where general emptiness rules override specific choice declarations.

This bug affects real-world use cases such as:
- Forms with "Not Selected" or "None" as meaningful options
- Optional fields where empty is a valid business state
- Database migrations where existing empty values need to be preserved
- Dropdown menus where the first option is intentionally empty

The Django forms layer actually handles this correctly - form fields with empty values in choices accept those values. This inconsistency between model and form validation adds to developer confusion.

Source code reference: `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/db/models/fields/__init__.py:807-827`

## Proposed Fix

```diff
--- a/django/db/models/fields/__init__.py
+++ b/django/db/models/fields/__init__.py
@@ -804,10 +804,11 @@ class Field(RegisterLookupMixin):
         if not self.editable:
             # Skip validation for non-editable fields.
             return

-        if self.choices is not None and value not in self.empty_values:
+        if self.choices is not None:
             for option_key, option_value in self.choices:
                 if isinstance(option_value, (list, tuple)):
                     # This is an optgroup, so look inside the group for
                     # options.
                     for optgroup_key, optgroup_value in option_value:
                         if value == optgroup_key:
                             return
                 elif value == option_key:
                     return
+            # Value not found in choices
             raise exceptions.ValidationError(
                 self.error_messages["invalid_choice"],
                 code="invalid_choice",
                 params={"value": value},
             )

+        # Only check blank/null if choices validation didn't already handle it
         if value is None and not self.null:
             raise exceptions.ValidationError(self.error_messages["null"], code="null")

         if not self.blank and value in self.empty_values:
             raise exceptions.ValidationError(self.error_messages["blank"], code="blank")
```