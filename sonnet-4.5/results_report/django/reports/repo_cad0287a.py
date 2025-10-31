#!/usr/bin/env python3
"""
Minimal reproduction of Django TemplateCommand --exclude bug.
This simulates the exact logic from django/core/management/templates.py lines 167-172.
"""

def filter_directories_without_exclude(dirs):
    """Filter directories when no --exclude option is provided"""
    result = dirs[:]
    options = {}  # No exclude option
    excluded_directories = [".git", "__pycache__"]

    for dirname in dirs[:]:
        if "exclude" not in options:
            if dirname.startswith(".") or dirname == "__pycache__":
                result.remove(dirname)
        elif dirname in excluded_directories:
            result.remove(dirname)

    return result

def filter_directories_with_exclude(dirs, user_excludes):
    """Filter directories when --exclude option IS provided"""
    result = dirs[:]
    options = {"exclude": user_excludes}  # Has exclude option
    excluded_directories = [".git", "__pycache__"] + user_excludes

    for dirname in dirs[:]:
        if "exclude" not in options:
            if dirname.startswith(".") or dirname == "__pycache__":
                result.remove(dirname)
        elif dirname in excluded_directories:
            result.remove(dirname)

    return result

# Test directories that would be encountered
test_dirs = ['.git', '.vscode', '.idea', '.DS_Store', '__pycache__', 'myapp', 'static', 'templates']

print("=" * 60)
print("Django TemplateCommand --exclude Bug Demonstration")
print("=" * 60)
print()
print(f"Test directories: {test_dirs}")
print()

# Scenario 1: No --exclude option
filtered_without = filter_directories_without_exclude(test_dirs[:])
print("Scenario 1: No --exclude option")
print(f"Remaining after filter: {filtered_without}")
print(f"Excluded: {[d for d in test_dirs if d not in filtered_without]}")
print()

# Scenario 2: With --exclude myexclude
filtered_with = filter_directories_with_exclude(test_dirs[:], ['myexclude'])
print("Scenario 2: With --exclude myexclude")
print(f"Remaining after filter: {filtered_with}")
print(f"Excluded: {[d for d in test_dirs if d not in filtered_with]}")
print()

print("=" * 60)
print("BUG: When using --exclude, dot directories are NOT excluded!")
print("=" * 60)
print()
print("Expected behavior: Both scenarios should exclude ALL dot directories")
print(f"  (.git, .vscode, .idea, .DS_Store) and __pycache__")
print()
print("Actual behavior:")
print(f"  - Without --exclude: Correctly excludes all dot dirs and __pycache__")
print(f"  - With --exclude: Only excludes .git and __pycache__, allows other dot dirs")
print()
print("Root cause: The if/elif logic at lines 168-172 makes filtering")
print("LESS restrictive when --exclude is used, which is counterintuitive.")