# Contributing Bug Status Updates

Help us track which bugs have been reported and validated by maintainers!

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

## Guidelines

1. **Include the GitHub link** - Add the issue or PR URL as evidence
2. **Be accurate** - Only mark as "valid" if maintainer explicitly confirmed it's a bug
3. **Keep it simple** - Just status and URL, nothing else needed

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