# Contributing Bug Status Updates

If you are a maintainer of a Python package we tested, you can help us confirm whether the bugs are real or not!

## How to Update Bug Status

1. **Fork this repository**
2. **Edit `status.json`** to add or update bug statuses
3. **Submit a Pull Request** with your changes

## Status Categories

- **`unknown`** - Not yet reported to maintainers or awaiting response
- **`valid`** - Maintainer confirmed this is a real bug
- **`invalid`** - Maintainer said this is not a bug / working as intended

## Format

Add an entry to `status.json` under the package name:

```json
"package_name": {
  "bug_report_file.md": {
    "status": "valid",
    "url": "https://github.com/org/repo/pull/123"
  }
}
```

If there is an issue/PR link, add it as evidence; if you're the maintainer, it's not needed, but it would be nice to, e.g., a PR link of the fix.

## Examples

### Valid Bug (with PR)
```json
"numpy": {
  "bug_report_numpy_random_wald_2025-08-18_05-03_x7k9.md": {
    "status": "valid",
    "url": "https://github.com/numpy/numpy/pull/12345"
  }
}
```

### Invalid Bug (Not a Bug)
```json
"python-dateutil": {
  "bug_report_python_dateutil_easter_2025-08-18_21-48_k9x2.md": {
    "status": "invalid",
    "url": "https://github.com/dateutil/dateutil/issues/8901"
  }
}
```

Thank you for helping validate these bugs!