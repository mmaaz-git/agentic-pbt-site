# Bug Report: TemplateCommand --exclude Makes Filtering Less Restrictive

**Target**: `django.core.management.templates.TemplateCommand.handle`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When using the `--exclude` option with Django's `startapp` or `startproject` commands, the directory filtering becomes LESS restrictive rather than MORE restrictive. Specifically, without `--exclude`, all directories starting with `.` are excluded, but WITH `--exclude`, only `.git` is excluded, allowing directories like `.vscode`, `.idea`, etc. to be copied.

## Property-Based Test

```python
from hypothesis import given, strategies as st


@given(st.sets(st.text(min_size=1, max_size=20), min_size=1, max_size=5))
def test_exclude_preserves_dot_directory_filtering(exclude_dirs):
    dirs_without_exclude = ['.git', '.vscode', '.idea', '__pycache__', 'mydir']
    dirs_with_exclude = dirs_without_exclude[:]

    excluded_directories = [".git", "__pycache__"] + list(exclude_dirs)

    filtered_without = []
    for dirname in dirs_without_exclude:
        if not (dirname.startswith(".") or dirname == "__pycache__"):
            filtered_without.append(dirname)

    filtered_with = []
    for dirname in dirs_with_exclude:
        if dirname not in excluded_directories:
            filtered_with.append(dirname)

    dot_dirs_without = [d for d in dirs_without_exclude if d.startswith('.') and d not in filtered_without]
    dot_dirs_with = [d for d in dirs_with_exclude if d.startswith('.') and d not in filtered_with]

    assert len(dot_dirs_with) >= len(dot_dirs_without), \
        f"Using --exclude should not make filtering LESS restrictive for dot directories"
```

**Failing input**: `exclude_dirs={'temp'}` - results in `.vscode` and `.idea` being included when they shouldn't be.

## Reproducing the Bug

```python
excluded_directories = [".git", "__pycache__", "myexclude"]
options_with_exclude = {"exclude": ["myexclude"]}
options_without_exclude = {}

dirs1 = ['.git', '.vscode', '__pycache__', 'mydir']
dirs2 = ['.git', '.vscode', '__pycache__', 'mydir']

for dirname in dirs1[:]:
    if "exclude" not in options_without_exclude:
        if dirname.startswith(".") or dirname == "__pycache__":
            dirs1.remove(dirname)

for dirname in dirs2[:]:
    if "exclude" not in options_with_exclude:
        if dirname.startswith(".") or dirname == "__pycache__":
            dirs2.remove(dirname)
    elif dirname in excluded_directories:
        dirs2.remove(dirname)

print(f"Without --exclude: {dirs1}")
print(f"With --exclude myexclude: {dirs2}")
```

Output:
```
Without --exclude: ['mydir']
With --exclude myexclude: ['.vscode', 'mydir']
```

Expected behavior: Both should exclude `.vscode` (and all other dot directories)

## Why This Is A Bug

The current logic creates an inconsistent user experience where adding `--exclude` to customize exclusions inadvertently makes the filtering less strict for dot directories. Users would reasonably expect that:

1. By default, all dot directories and `__pycache__` are excluded
2. Using `--exclude foo` would ADDITIONALLY exclude `foo`, not REPLACE the default dot-directory filtering

The root cause is in lines 167-172 of `/django/core/management/templates.py`:

```python
for dirname in dirs[:]:
    if "exclude" not in options:
        if dirname.startswith(".") or dirname == "__pycache__":
            dirs.remove(dirname)
    elif dirname in excluded_directories:
        dirs.remove(dirname)
```

## Fix

```diff
--- a/django/core/management/templates.py
+++ b/django/core/management/templates.py
@@ -166,9 +166,8 @@ class TemplateCommand(BaseCommand):

             for dirname in dirs[:]:
-                if "exclude" not in options:
-                    if dirname.startswith(".") or dirname == "__pycache__":
-                        dirs.remove(dirname)
-                elif dirname in excluded_directories:
+                if dirname.startswith(".") or dirname in excluded_directories:
                     dirs.remove(dirname)

             for filename in files:
```

This fix ensures that:
1. All dot directories are always excluded (consistent behavior)
2. The `excluded_directories` list (which includes `.git`, `__pycache__`, and user-specified directories) is also checked
3. The filtering is consistent whether or not `--exclude` is used