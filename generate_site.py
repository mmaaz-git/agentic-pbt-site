#!/usr/bin/env python3
import os
import json
import re
import csv
from pathlib import Path

def parse_opus_bug_report(file_path):
    """Parse a bug report from opus-4.1 results"""
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

def parse_sonnet_bug_report(file_path, call_mappings, package_name, bug_report_to_info, enhanced_reports, report_package_dir):
    """Parse a bug report from sonnet-4.5 results_verify"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract title
        title_match = re.search(r'^# Bug Report: (.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Unknown"

        # Extract metadata from verify bug report
        target_match = re.search(r'\*\*Target\*\*:\s*`?([^`\n]+)`?', content)
        bug_type_match = re.search(r'\*\*Bug Type\*\*:\s*(\w+)', content)
        date_match = re.search(r'\*\*Date\*\*:\s*(\d{4}-\d{2}-\d{2})', content)

        # Extract summary
        summary_match = re.search(r'## Summary\s*\n\n(.+?)(?=\n##|\n\*\*|\Z)', content, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ""

        # Clean up summary - remove excessive newlines
        summary = re.sub(r'\n+', ' ', summary)

        # Get call_id and category from bug_report_to_info mapping
        verify_info = bug_report_to_info.get(file_path.name, {})
        call_id = verify_info.get('call_id')
        category = verify_info.get('category', 'Unknown')

        # Get analysis file path
        analysis_path = None
        if call_id:
            analysis_path = f"sonnet-4.5/results_verify/{package_name}/verifications/analysis_{call_id}.txt"

        # For BUG category, get severity from enhanced report
        severity = None

        if category == 'BUG':
            # Check if enhanced report exists
            enhanced_report_name = enhanced_reports.get(file_path.name)
            if not enhanced_report_name:
                # BUG without enhanced report - drop it
                return None

            # Read enhanced report to get severity
            enhanced_report_file = report_package_dir / 'reports' / enhanced_report_name
            if enhanced_report_file.exists():
                try:
                    with open(enhanced_report_file, 'r', encoding='utf-8') as f:
                        enhanced_content = f.read()
                    severity_match = re.search(r'\*\*Severity\*\*:\s*([^\n]+)', enhanced_content)
                    if severity_match:
                        severity = severity_match.group(1).strip()
                except Exception as e:
                    print(f"Error reading enhanced report {enhanced_report_file}: {e}")

            if not severity:
                # BUG without severity - drop it
                return None

        return {
            'title': title,
            'target': target_match.group(1) if target_match else "Unknown",
            'severity': severity if severity else "N/A",  # N/A for non-BUG categories
            'bug_type': bug_type_match.group(1) if bug_type_match else "Unknown",
            'date': date_match.group(1) if date_match else "Unknown",
            'summary': summary[:500],
            'package': package_name,
            'file_name': file_path.name,
            'file_path': str(file_path).replace(str(Path.cwd()) + '/', ''),
            'call_id': call_id,
            'category': category,
            'analysis_path': analysis_path
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def generate_opus_data():
    """Generate data for opus-4.1 results"""
    print("Generating Opus 4.1 data...")
    results_dir = Path('opus-4.1')
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
                # Update results to opus-4.1 if needed
                filepath = filepath.replace('results/', 'opus-4.1/')
                filepath = filepath.replace('reports/', 'opus-4.1/')
                scores[filepath] = {
                    'total_score': int(row['score']) if row['score'] else None,
                    'obviousness': int(row['obviousness']) if row['obviousness'] else None,
                    'input_reasonableness': int(row['input_reasonableness']) if row['input_reasonableness'] else None,
                    'maintainer_defensibility': int(row['maintainer_defensibility']) if row['maintainer_defensibility'] else None,
                    'valid': row.get('valid', ''),
                    'reportable': row.get('reportable', ''),
                    'response': row.get('response', '')
                }
                scores_loaded += 1

    # Find all bug report markdown files
    matched_scores = 0
    for md_file in results_dir.glob('*/bug_reports/*.md'):
        report = parse_opus_bug_report(md_file)
        if report:
            # Add score data if available
            file_path_key = report['file_path']
            if file_path_key in scores:
                report['score'] = scores[file_path_key]
                matched_scores += 1
            else:
                report['score'] = None

            # Add bug status if available
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
        severity = report['severity']
        stats['severity_counts'][severity] = stats['severity_counts'].get(severity, 0) + 1

        bug_type = report['bug_type']
        stats['bug_type_counts'][bug_type] = stats['bug_type_counts'].get(bug_type, 0) + 1

        package = report['package']
        stats['package_counts'][package] = stats['package_counts'].get(package, 0) + 1

    output = {
        'reports': all_reports,
        'stats': stats
    }

    with open('data-opus-4.1.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"Parsed {len(all_reports)} bug reports from {stats['packages']} packages")
    print(f"Loaded {scores_loaded} scores from CSV")
    print(f"Matched {matched_scores} scores to bug reports")
    print(f"Data saved to data-opus-4.1.json")

    if invalid_entries:
        print("\n❌ WARNING: Invalid entries found in status.json!")
        print("The following files do not exist in the opus-4.1 directory:")
        for entry in invalid_entries:
            print(f"  - {entry}")

    print("\nSeverity distribution:")
    for severity, count in sorted(stats['severity_counts'].items()):
        print(f"  {severity}: {count}")

def generate_sonnet_data():
    """Generate data for sonnet-4.5 results_verify"""
    print("\nGenerating Sonnet 4.5 data...")
    verify_dir = Path('sonnet-4.5/results_verify')
    report_dir = Path('sonnet-4.5/results_report')
    all_reports = []

    if not verify_dir.exists():
        print(f"Warning: {verify_dir} does not exist, skipping sonnet data generation")
        return

    # Load bug statuses (shared with opus or separate)
    bug_statuses = {}
    if Path('status.json').exists():
        with open('status.json', 'r', encoding='utf-8') as f:
            bug_statuses = json.load(f)

    # Process each package in results_verify
    for package_dir in verify_dir.iterdir():
        if not package_dir.is_dir() or package_dir.name.startswith('.'):
            continue

        package_name = package_dir.name
        bug_reports_dir = package_dir / 'bug_reports'
        call_mappings_file = package_dir / 'call_mappings.jsonl'

        if not bug_reports_dir.exists():
            continue

        # Load call_mappings from results_verify
        verify_call_mappings = []
        if call_mappings_file.exists():
            with open(call_mappings_file, 'r', encoding='utf-8') as f:
                for line in f:
                    verify_call_mappings.append(json.loads(line))

        # Build mapping from bug_report filename to call_id and category
        bug_report_to_info = {}
        for mapping in verify_call_mappings:
            bug_report_name = mapping['bug_report']
            call_id = mapping['call_id']
            category = mapping.get('verification_category', 'Unknown')
            bug_report_to_info[bug_report_name] = {
                'call_id': call_id,
                'category': category
            }

        # Load enhanced report paths from results_report if available
        report_package_dir = report_dir / package_name
        report_call_mappings = []
        if report_package_dir.exists():
            report_call_mappings_file = report_package_dir / 'call_mappings.jsonl'
            if report_call_mappings_file.exists():
                with open(report_call_mappings_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        report_call_mappings.append(json.loads(line))

        # Build mapping from bug_report to enhanced_report
        enhanced_reports = {}
        for mapping in report_call_mappings:
            if 'enhanced_report' in mapping:
                enhanced_reports[mapping['bug_report']] = mapping['enhanced_report']

        # Parse bug reports from results_verify
        for md_file in bug_reports_dir.glob('*.md'):
            report = parse_sonnet_bug_report(md_file, report_call_mappings, package_name, bug_report_to_info, enhanced_reports, report_package_dir)
            if report:
                # Add bug status if available
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
        'category_counts': {},
        'bug_type_counts': {},
        'package_counts': {}
    }

    for report in all_reports:
        severity = report['severity']
        stats['severity_counts'][severity] = stats['severity_counts'].get(severity, 0) + 1

        category = report['category']
        stats['category_counts'][category] = stats['category_counts'].get(category, 0) + 1

        bug_type = report['bug_type']
        stats['bug_type_counts'][bug_type] = stats['bug_type_counts'].get(bug_type, 0) + 1

        package = report['package']
        stats['package_counts'][package] = stats['package_counts'].get(package, 0) + 1

    output = {
        'reports': all_reports,
        'stats': stats
    }

    with open('data-sonnet-4.5.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"Parsed {len(all_reports)} bug reports from {stats['packages']} packages")
    print(f"Data saved to data-sonnet-4.5.json")

    print("\nSeverity distribution:")
    for severity, count in sorted(stats['severity_counts'].items()):
        print(f"  {severity}: {count}")

    print("\nCategory distribution:")
    for category, count in sorted(stats['category_counts'].items()):
        print(f"  {category}: {count}")

def main():
    """Generate data for both opus and sonnet results"""
    generate_opus_data()
    generate_sonnet_data()

if __name__ == "__main__":
    main()
