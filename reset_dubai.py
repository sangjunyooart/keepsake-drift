#!/usr/bin/env python3
"""
Reset database with 6 Dubai-themed memory minds
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = './data/keepsake.sqlite'

# 6 Dubai-themed minds
DUBAI_MINDS = [
    {
        'mind_id': 1,
        'name': 'souk',
        'display_name': 'Gold Souk Merchant',
        'description': 'Traditional merchant from the golden era of Dubai souks',
        'initial_drift': 'The weight of gold chains in my hands. The sound of bargaining voices echoing through narrow passages. Before the towers rose, we measured wealth in different ways.',
    },
    {
        'mind_id': 2,
        'name': 'tower',
        'display_name': 'Tower Builder',
        'description': 'Construction worker who built the Burj Khalifa',
        'initial_drift': 'Steel and concrete rising floor by floor. Each level higher than the last. We built the tallest dream the world had ever seen, one beam at a time.',
    },
    {
        'mind_id': 3,
        'name': 'desert',
        'display_name': 'Desert Guide',
        'description': 'Bedouin guide who knows the old paths',
        'initial_drift': 'The dunes shift but the stars remain. My grandfather taught me to read the sand like a map. Now the city lights drown out half the sky.',
    },
    {
        'mind_id': 4,
        'name': 'pearl',
        'display_name': 'Pearl Diver',
        'description': 'Pearl diver from before the oil era',
        'initial_drift': 'Diving deep into green water. Holding my breath. The pearls we brought up fed entire families. Before black gold replaced white pearls.',
    },
    {
        'mind_id': 5,
        'name': 'startup',
        'display_name': 'Startup Founder',
        'description': 'Tech entrepreneur in Dubai Internet City',
        'initial_drift': 'Pitching to investors in glass towers. Building apps that connect continents. This city rewards those who bet on tomorrow instead of yesterday.',
    },
    {
        'mind_id': 6,
        'name': 'expat',
        'display_name': 'Expat Family',
        'description': 'Expatriate family making Dubai their home',
        'initial_drift': 'Unpacking boxes in a new apartment. Learning to navigate malls the size of cities. Building a life between cultures, between old home and new.',
    },
]

def reset_database():
    """Delete and recreate database with Dubai minds"""

    # Remove old database
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"✅ Removed old database: {DB_PATH}")

    # Create new database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create minds table
    cursor.execute("""
        CREATE TABLE minds (
            mind_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create drift_memory table
    cursor.execute("""
        CREATE TABLE drift_memory (
            drift_id INTEGER PRIMARY KEY AUTOINCREMENT,
            mind_id INTEGER NOT NULL,
            tick_id INTEGER NOT NULL DEFAULT 0,
            parent_drift_id INTEGER,
            version INTEGER NOT NULL DEFAULT 0,
            drift_text TEXT NOT NULL,
            drift_text_ar TEXT,
            summary_text TEXT,
            summary_text_ar TEXT,
            delta_json TEXT,
            ar_patch_json TEXT,
            params_json TEXT,
            keepsake_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (mind_id) REFERENCES minds(mind_id),
            FOREIGN KEY (parent_drift_id) REFERENCES drift_memory(drift_id)
        )
    """)

    # Create ticks table
    cursor.execute("""
        CREATE TABLE ticks (
            tick_id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            params_json TEXT
        )
    """)

    # Create user_fragments table
    cursor.execute("""
        CREATE TABLE user_fragments (
            fragment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            mind_id INTEGER NOT NULL,
            fragment_text TEXT NOT NULL,
            fragment_text_ar TEXT,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (mind_id) REFERENCES minds(mind_id)
        )
    """)

    # Insert Dubai minds
    for mind in DUBAI_MINDS:
        cursor.execute("""
            INSERT INTO minds (mind_id, name, display_name, description)
            VALUES (?, ?, ?, ?)
        """, (mind['mind_id'], mind['name'], mind['display_name'], mind['description']))

        # Insert initial drift for each mind
        cursor.execute("""
            INSERT INTO drift_memory (
                mind_id, tick_id, version, drift_text,
                summary_text, params_json, keepsake_text
            ) VALUES (?, 0, 0, ?, ?, '{}', ?)
        """, (
            mind['mind_id'],
            mind['initial_drift'],
            mind['display_name'] + ' remembers',
            mind['initial_drift'][:100] + '...'
        ))

        print(f"✅ Created mind: {mind['display_name']} ({mind['name']})")

    # Insert initial tick
    cursor.execute("""
        INSERT INTO ticks (tick_id, params_json)
        VALUES (0, '{"note": "Initial Dubai setup"}')
    """)

    conn.commit()
    conn.close()

    print(f"\n✅ Database reset complete!")
    print(f"✅ Created 6 Dubai-themed minds")
    print(f"✅ Database location: {DB_PATH}")

if __name__ == '__main__':
    reset_database()
