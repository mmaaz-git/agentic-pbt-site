#!/usr/bin/env python3
"""
Process contractor review files and extract structured data.

Combines contractors_review_2.json, contractors_review_3.json, and contractors_review_4.json
into a single processed file with global_key as the main key and call_id extracted.
Also looks up bug report filenames from results_report call_mappings.
"""

import json
from pathlib import Path

def extract_call_id_from_global_key(global_key):
    """Extract call_id from global_key format: html_bug_report_{call_id}_{report_id}"""
    parts = global_key.split('_')
    if len(parts) >= 4:
        return parts[3]  # The call_id is the 4th part
    return None

def load_call_id_to_filename_mapping():
    """Load mapping from call_id to bug_report filename from results_report and opus manual mapping."""
    mapping = {}

    # Load Sonnet mappings from results_report
    report_dir = Path('sonnet-4.5/results_report')
    if report_dir.exists():
        for call_mapping_file in report_dir.rglob('call_mappings.jsonl'):
            package_name = call_mapping_file.parent.name
            with open(call_mapping_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        call_id = data.get('call_id')
                        bug_report = data.get('bug_report')
                        if call_id and bug_report:
                            mapping[call_id] = {
                                'bug_report': bug_report,
                                'package': package_name
                            }
        print(f"Loaded {len(mapping)} Sonnet call_id → filename mappings from results_report")
    else:
        print(f"Warning: {report_dir} does not exist")

    # Load Opus manual mappings
    opus_mapping_file = Path('opus_human_reviews_mapping.json')
    if opus_mapping_file.exists():
        with open(opus_mapping_file, 'r', encoding='utf-8') as f:
            opus_mappings = json.load(f)
            mapping.update(opus_mappings)
        print(f"Loaded {len(opus_mappings)} Opus call_id → filename mappings from manual file")

    return mapping

def process_contractor_reviews():
    """Process all contractor review files and create structured output."""

    # Load call_id to filename mapping
    call_id_mapping = load_call_id_to_filename_mapping()

    # Read all contractor review files
    all_reviews = []
    for i in [2, 3, 4]:
        review_file = Path(f'contractors_review_{i}.json')
        if review_file.exists():
            with open(review_file, 'r', encoding='utf-8') as f:
                reviews = json.load(f)
                all_reviews.extend(reviews)
                print(f"Loaded {len(reviews)} reviews from {review_file}")

    print(f"\nTotal reviews loaded: {len(all_reviews)}")

    # Process reviews by global_key
    processed = {}

    for review in all_reviews:
        global_key = review.get('global_key')
        if not global_key:
            continue

        # Extract call_id from global_key
        call_id = extract_call_id_from_global_key(global_key)

        # Look up bug report filename and package
        bug_report = None
        package = None
        if call_id and call_id in call_id_mapping:
            bug_report = call_id_mapping[call_id]['bug_report']
            package = call_id_mapping[call_id]['package']

        # Initialize entry if first time seeing this global_key
        if global_key not in processed:
            processed[global_key] = {
                'call_id': call_id,
                'bug_report': bug_report,
                'package': package,
                'url': review.get('url', ''),
                'reviews': []
            }

        # Process each contractor's vote
        for vote in review.get('votes', []):
            q1 = vote.get('Q1', {})
            q2 = vote.get('Q2', {})

            review_entry = {
                'rater_id': vote.get('rater_id'),
                'q1_vote': q1.get('vote'),
                'q1_confidence': q1.get('confidence'),
                'q2_vote': q2.get('vote'),
                'q2_confidence': q2.get('confidence'),
                'comment': vote.get('comments', '')
            }

            processed[global_key]['reviews'].append(review_entry)

    # Write processed data
    output_file = Path('contractor_reviews_processed.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"\nProcessed {len(processed)} unique bug reports")
    print(f"Output written to {output_file}")

    # Print some statistics
    print("\nStatistics:")
    total_reviews = sum(len(v['reviews']) for v in processed.values())
    print(f"  Total contractor reviews: {total_reviews}")
    print(f"  Average reviews per bug: {total_reviews / len(processed):.1f}")

    # Count call_id distribution
    call_ids = [v['call_id'] for v in processed.values() if v['call_id']]
    print(f"  Unique call_ids: {len(set(call_ids))}")

    # Count how many have bug_report filenames
    with_filename = [v for v in processed.values() if v['bug_report']]
    without_filename = [v for v in processed.values() if not v['bug_report']]
    print(f"  With bug_report filename: {len(with_filename)}")
    print(f"  Without bug_report filename: {len(without_filename)}")

    return processed

if __name__ == '__main__':
    processed = process_contractor_reviews()

    # Show a sample entry
    if processed:
        print("\nSample entry:")
        sample_key = list(processed.keys())[0]
        sample = processed[sample_key]
        print(f"Global key: {sample_key}")
        print(f"Call ID: {sample['call_id']}")
        print(f"Number of reviews: {len(sample['reviews'])}")
        if sample['reviews']:
            print(f"First review Q1: {sample['reviews'][0]['q1_vote']}")
            print(f"Comment preview: {sample['reviews'][0]['comment'][:100]}...")
