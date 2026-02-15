#!/usr/bin/env python3
"""
Expand Dubai drift texts with longer, richer narratives
"""
import sqlite3

DB_PATH = './data/keepsake.sqlite'

EXPANDED_DRIFTS = {
    'souk': '''The weight of gold chains in my hands. The sound of bargaining voices echoing through narrow passages. Before the towers rose, we measured wealth in different ways.

In the old souk, every transaction was a conversation. You would never just buy—you would sit, drink tea, discuss the craftsmanship. The gold spoke its own language, passed down through generations of artisans whose fingers knew the metal like family.

Now the new malls glitter with the same gold, but something has changed in the air. The bargaining is quieter, more polite. The tea comes in paper cups. Yet when I close my eyes, I can still hear the old rhythm—the clink of chains, the murmur of prices, the weight of tradition in every gram.''',

    'tower': '''Steel and concrete rising floor by floor. Each level higher than the last. We built the tallest dream the world had ever seen, one beam at a time.

I remember the early mornings, the city still dark below us. We were building something impossible—828 meters into the sky. Every day we climbed higher, past where eagles fly, past where you could see the curvature of the earth.

The heat was relentless, the work dangerous. But we were making history with our hands. When the last spire was placed, I looked down at the city spread beneath us like a circuit board of light, and I knew: we had changed the skyline forever. Some built with stone in ancient times. We built with steel and faith that the impossible was just another day of work.''',

    'desert': '''The dunes shift but the stars remain. My grandfather taught me to read the sand like a map. Now the city lights drown out half the sky.

Once, we knew every water source within a hundred kilometers. The desert was not empty—it was full of signs if you knew how to look. A certain plant growing meant water beneath. The way the sand gathered against a rock told you which way the wind would blow tomorrow.

The old paths are still there, beneath the highways and glass towers. Sometimes I drive out past the city limits, past where the streetlights end, and I remember. The silence of the desert is different now—contaminated by the hum of distant generators, the glow of development on every horizon. But the stars still remember the old routes, even if we have forgotten.''',

    'pearl': '''Diving deep into green water. Holding my breath. The pearls we brought up fed entire families. Before black gold replaced white pearls.

The old men still talk about the diving seasons—how we would go down, down, down until our lungs burned and the pressure made our ears ring. You learned to stay calm in the depths, to move efficiently, to trust your body to know when it was time to surface.

We were wealthy once, in a different way. Not from what we extracted from the earth, but from what we found in the sea. The pearls were small fortunes, each one unique, each one earned through held breath and aching muscles. When oil came, the diving stopped almost overnight. Now the young ones barely remember that we were a nation of pearl divers before we were a nation of towers.''',

    'startup': '''Pitching to investors in glass towers. Building apps that connect continents. This city rewards those who bet on tomorrow instead of yesterday.

Dubai taught me to think bigger. Not just regional, but global. Not just profitable, but transformative. Every pitch meeting is a test of vision—can you make them see the future you see? Can you compress years of work into fifteen slides?

The ecosystem here is electric. You sit in a café in Dubai Internet City and overhear conversations in six languages, all about the next big thing. Funding flows to those who can articulate the impossible and make it sound inevitable. Some days I feel like we are building castles in the air. Other days I remember: this entire city was built on vision that seemed impossible fifty years ago. We are just continuing the tradition, one line of code at a time.''',

    'expat': '''Unpacking boxes in a new apartment. Learning to navigate malls the size of cities. Building a life between cultures, between old home and new.

The first year was hardest. Everything familiar was thousands of miles away—the foods, the weather, the casual friendships built over decades. Here, everyone is from somewhere else, which means everyone understands displacement, but also that everything feels temporary.

We celebrate two sets of holidays now. We switch between languages mid-sentence. Our children will grow up thinking this multiplicity is normal—that home can be a concept rather than a fixed location. Sometimes I miss the simplicity of belonging to one place. But then I see how my daughter navigates three cultures with ease, how she builds bridges I never could, and I think: this is the real wealth we are accumulating. Not in bank accounts, but in the ability to be at home anywhere.'''
}

def expand_drifts():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for mind_key, drift_text in EXPANDED_DRIFTS.items():
        # Get mind_id
        mind_id = cursor.execute(
            "SELECT mind_id FROM minds WHERE mind_key = ?",
            (mind_key,)
        ).fetchone()[0]

        # Update drift text
        cursor.execute("""
            UPDATE drift_memory
            SET drift_text = ?
            WHERE mind_id = ? AND version = 0
        """, (drift_text, mind_id))

        print(f"✅ Expanded drift for: {mind_key}")

    conn.commit()
    conn.close()
    print("\n✅ All drift texts expanded!")

if __name__ == '__main__':
    expand_drifts()
