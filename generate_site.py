#!/usr/bin/env python3
import os
import json
import re
import csv
from pathlib import Path

def parse_bug_report(file_path):
    """Parse a single bug report markdown file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract title
        title_match = re.search(r'^# Bug Report: (.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Unknown"

        # Extract metadata
        target_match = re.search(r'\*\*Target\*\*:\s*`?([^`\n]+)`?', content)
        severity_match = re.search(r'\*\*Severity\*\*:\s*(\w+)', content)
        bug_type_match = re.search(r'\*\*Bug Type\*\*:\s*(\w+)', content)
        date_match = re.search(r'\*\*Date\*\*:\s*(\d{4}-\d{2}-\d{2})', content)

        # Extract summary
        summary_match = re.search(r'## Summary\s*\n\n(.+?)(?=\n##|\n\*\*|\Z)', content, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ""

        # Clean up summary - remove excessive newlines
        summary = re.sub(r'\n+', ' ', summary)

        # Get package name from file path
        package = file_path.parent.parent.name

        return {
            'title': title,
            'target': target_match.group(1) if target_match else "Unknown",
            'severity': severity_match.group(1) if severity_match else "Unknown",
            'bug_type': bug_type_match.group(1) if bug_type_match else "Unknown",
            'date': date_match.group(1) if date_match else "Unknown",
            'summary': summary[:500],  # Limit summary length
            'package': package,
            'file_name': file_path.name,
            'file_path': str(file_path).replace(str(Path.cwd()) + '/', '')
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def main():
    """Parse all bug reports and generate JSON data"""
    results_dir = Path('reports')
    all_reports = []

    # Load bug statuses and validate entries
    bug_statuses = {}
    invalid_entries = []
    if Path('status.json').exists():
        with open('status.json', 'r', encoding='utf-8') as f:
            bug_statuses = json.load(f)
        
        # Validate that all entries in status.json correspond to actual files
        for package, files in bug_statuses.items():
            package_dir = results_dir / package / 'bug_reports'
            for filename in files:
                file_path = package_dir / filename
                if not file_path.exists():
                    invalid_entries.append(f"{package}/{filename}")

    # Load scores from CSV
    scores = {}
    scores_loaded = 0
    if Path('scores.csv').exists():
        with open('scores.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract filename from the path in CSV
                filepath = row['file']
                # Remove 'clean/' prefix if present
                filepath = filepath.replace('clean/', '')
                # Update results to reports if needed
                filepath = filepath.replace('results/', 'reports/')
                scores[filepath] = {
                    'total_score': int(row['score']) if row['score'] else None,
                    'obviousness': int(row['obviousness']) if row['obviousness'] else None,
                    'input_reasonableness': int(row['input_reasonableness']) if row['input_reasonableness'] else None,
                    'maintainer_defensibility': int(row['maintainer_defensibility']) if row['maintainer_defensibility'] else None,
                    'valid': row.get('valid', ''),
                    'reportable': row.get('reportable', ''),
                    'response': row.get('response', '')  # Include Claude's detailed analysis
                }
                scores_loaded += 1

    # Find all bug report markdown files
    matched_scores = 0
    for md_file in results_dir.glob('*/bug_reports/*.md'):
        report = parse_bug_report(md_file)
        if report:
            # Add score data if available
            file_path_key = report['file_path']
            if file_path_key in scores:
                report['score'] = scores[file_path_key]
                matched_scores += 1
            else:
                report['score'] = None

            # Add bug status if available (check nested structure)
            package_name = report['package']
            file_name = report['file_name']
            
            if package_name in bug_statuses and file_name in bug_statuses[package_name]:
                report['bug_status'] = bug_statuses[package_name][file_name]
            else:
                report['bug_status'] = {
                    'status': 'unknown',
                    'url': ''
                }

            all_reports.append(report)

    # Sort by package and then by title
    all_reports.sort(key=lambda x: (x['package'], x['title']))

    # Create summary statistics
    stats = {
        'total_reports': len(all_reports),
        'packages': len(set(r['package'] for r in all_reports)),
        'severity_counts': {},
        'bug_type_counts': {},
        'package_counts': {}
    }

    for report in all_reports:
        # Count severities
        severity = report['severity']
        stats['severity_counts'][severity] = stats['severity_counts'].get(severity, 0) + 1

        # Count bug types
        bug_type = report['bug_type']
        stats['bug_type_counts'][bug_type] = stats['bug_type_counts'].get(bug_type, 0) + 1

        # Count per package
        package = report['package']
        stats['package_counts'][package] = stats['package_counts'].get(package, 0) + 1

    # Save to JSON file
    output = {
        'reports': all_reports,
        'stats': stats
    }

    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"Parsed {len(all_reports)} bug reports from {stats['packages']} packages")
    print(f"Loaded {scores_loaded} scores from CSV")
    print(f"Matched {matched_scores} scores to bug reports")
    print(f"Data saved to data.json")
    
    # Check for invalid entries in status.json
    if invalid_entries:
        print("\n❌ ERROR: Invalid entries found in status.json!")
        print("The following files do not exist in the reports directory:")
        for entry in invalid_entries:
            print(f"  - {entry}")
        print("\nPlease fix status.json to only reference existing bug report files.")
        print("Check the reports/ directory for the correct filenames.")
        exit(1)  # Exit with error code to fail CI

    # Print summary
    print("\nSeverity distribution:")
    for severity, count in sorted(stats['severity_counts'].items()):
        print(f"  {severity}: {count}")

    print("\nBug type distribution:")
    for bug_type, count in sorted(stats['bug_type_counts'].items()):
        print(f"  {bug_type}: {count}")

if __name__ == "__main__":
    main()