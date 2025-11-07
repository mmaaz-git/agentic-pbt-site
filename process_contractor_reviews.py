#!/usr/bin/env python3
"""
Process contractor review files and extract structured data.

Combines contractors_review_2.json, contractors_review_3.json, and contractors_review_4.json
into a single processed file with global_key as the main key and call_id extracted.
"""

import json
from pathlib import Path

def extract_call_id_from_global_key(global_key):
    """Extract call_id from global_key format: html_bug_report_{call_id}_{report_id}"""
    parts = global_key.split('_')
    if len(parts) >= 4:
        return parts[3]  # The call_id is the 4th part
    return None

def process_contractor_reviews():
    """Process all contractor review files and create structured output."""

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

        # Initialize entry if first time seeing this global_key
        if global_key not in processed:
            processed[global_key] = {
                'call_id': call_id,
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
