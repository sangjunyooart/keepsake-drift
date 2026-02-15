#!/usr/bin/env python3
"""
Seed initial drift texts for Dubai minds
"""
import sqlite3

DB_PATH = './data/keepsake.sqlite'

INITIAL_DRIFTS = {
    'souk': {
        'drift_text': 'The weight of gold chains in my hands. The sound of bargaining voices echoing through narrow passages. Before the towers rose, we measured wealth in different ways.',
        'summary_text': 'Gold Souk memories',
    },
    'tower': {
        'drift_text': 'Steel and concrete rising floor by floor. Each level higher than the last. We built the tallest dream the world had ever seen, one beam at a time.',
        'summary_text': 'Building the Burj',
    },
    'desert': {
        'drift_text': 'The dunes shift but the stars remain. My grandfather taught me to read the sand like a map. Now the city lights drown out half the sky.',
        'summary_text': 'Desert paths',
    },
    'pearl': {
        'drift_text': 'Diving deep into green water. Holding my breath. The pearls we brought up fed entire families. Before black gold replaced white pearls.',
        'summary_text': 'Pearl diving days',
    },
    'startup': {
        'drift_text': 'Pitching to investors in glass towers. Building apps that connect continents. This city rewards those who bet on tomorrow instead of yesterday.',
        'summary_text': 'Startup dreams',
    },
    'expat': {
        'drift_text': 'Unpacking boxes in a new apartment. Learning to navigate malls the size of cities. Building a life between cultures, between old home and new.',
        'summary_text': 'Finding home',
    },
}

def seed_drifts():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for mind_key, drift_data in INITIAL_DRIFTS.items():
        # Get mind_id
        mind_id = cursor.execute(
            "SELECT mind_id FROM minds WHERE mind_key = ?",
            (mind_key,)
        ).fetchone()[0]

        # Insert initial drift
        cursor.execute("""
            INSERT INTO drift_memory (
                mind_id, tick_id, version,
                drift_text, summary_text,
                params_json
            ) VALUES (?, 0, 0, ?, ?, '{}')
        """, (
            mind_id,
            drift_data['drift_text'],
            drift_data['summary_text']
        ))

        print(f"✅ Added initial drift for: {mind_key}")

    conn.commit()
    conn.close()
    print("\n✅ All initial drifts seeded!")

if __name__ == '__main__':
    seed_drifts()
