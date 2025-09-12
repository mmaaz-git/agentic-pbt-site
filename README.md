# Python Bug Reports from Property-Based Testing

This repository contains bug reports discovered through automated property-based testing of popular Python packages.

## View the Reports

Visit the [GitHub Pages site](https://[your-username].github.io/agentic-pbt-site/) to browse all bug reports.

## Statistics

- **Total Reports**: 984
- **Packages Tested**: 95
- **Severity Distribution**: 105 High, 725 Medium, 154 Low
- **Bug Types**: 441 Logic, 343 Contract, 200 Crash

## Repository Structure

- `index.html` - Main webpage for browsing reports
- `bug_reports_data.json` - Parsed bug report data
- `results/` - Original bug report markdown files organized by package
- `parse_bug_reports.py` - Script to parse markdown files and generate JSON data

## Running Locally

```bash
# Parse bug reports (if needed)
python3 parse_bug_reports.py

# Start local server
python3 -m http.server 8000

# Visit http://localhost:8000
```

## Deployment

This site is designed to be deployed on GitHub Pages. Simply push to the main branch and GitHub will automatically serve the site.