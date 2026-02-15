#!/usr/bin/env python3
"""
Set Dubai-themed initial memories for the 6 original temporalities
"""
import sqlite3

DB_PATH = './data/keepsake.sqlite'

# Dubai memories mapped to original temporalities
DUBAI_MEMORIES = {
    'human': '''The weight of gold chains in my hands. The sound of bargaining voices echoing through narrow passages. Before the towers rose, we measured wealth in different ways.

In the old souk, every transaction was a conversation. You would never just buy—you would sit, drink tea, discuss the craftsmanship. The gold spoke its own language, passed down through generations of artisans whose fingers knew the metal like family.

Now the new malls glitter with the same gold, but something has changed in the air. The bargaining is quieter, more polite. The tea comes in paper cups. Yet when I close my eyes, I can still hear the old rhythm—the clink of chains, the murmur of prices, the weight of tradition in every gram.''',

    'liminal': '''The threshold between old Dubai and new. Standing at the edge of the creek, watching abras cross the water. On one side, the heritage district with its wind towers. On the other, glass skyscrapers reflecting the afternoon sun.

This is where the city holds its breath between past and future. The call to prayer echoes from minarets even as construction cranes pivot overhead. Tourists photograph the old while residents navigate the new. Everyone passes through, no one quite belongs to either shore.

I take the water taxi across. Five dirhams, three minutes, fifty years of transformation compressed into the span of brown water between here and there.''',

    'environment': '''The heat arrives in waves, bending the air above black asphalt. By midday, the city empties into air-conditioned interiors. Only the construction workers remain outside, their shirts dark with sweat, seeking shade that doesn't exist.

The desert was always here, beneath the pavement and between the buildings. When the wind shifts, sand appears overnight on balconies and windshields, a reminder. The humidity from the Gulf makes breathing feel like drinking. Palm trees line the highways, each one fed by hidden pipes drawing water from plants that turn seawater fresh.

After sunset, the temperature drops just enough. People emerge. The corniche fills with walkers. The city exhales.''',

    'digital': '''The wifi reaches everywhere now—shopping malls, metro stations, even the beach. A thousand signals overlay the physical city with invisible pathways. Order food, call a car, pay bills, all from screens that glow in the dark.

Dubai Internet City hums with server farms. Fiber optic cables snake beneath the streets carrying data at light speed. The smart city dashboard tracks traffic flow, energy use, every meter of infrastructure converted to readable numbers. Cameras watch intersections, algorithms optimize routes, the city becomes its own network.

But sometimes the signal drops. For a moment, the invisible city flickers. You look up from your phone and remember there's a physical world, one that exists whether connected or not.''',

    'infrastructure': '''Water pipes buried beneath sand. Power lines strung between towers. Sewage treatment plants at the city's edge processing waste from two million toilets. The stuff no one sees but everyone needs.

They built an entire city from nothing. Dredged islands from the seabed. Pumped desalinated water upward through hundreds of kilometers of pipe. Laid fiber optic cable, electrical grid, gas lines—all the hidden veins that keep a city alive. Every day, trucks collect garbage from buildings, drive it to distant landfills, return empty to collect more.

When the infrastructure fails—a water main break, a power outage—the city suddenly becomes visible. People remember: this miracle of glass and steel depends on pipes and wires, on systems working perfectly, invisibly, all the time.''',

    'more_than_human': '''The falcons still hunt despite the city. Trained birds in climate-controlled mews, flown in the desert beyond the sprawl. An ancient partnership persists: human and raptor working as one, reading the wind together.

Camels pace in heritage villages, rented for tourist photos. They remember when this was all theirs. In the mangroves along the coast, herons nest among the roots, filtering saltwater through ancient biology. Insects spiral around streetlights, their navigation broken by electric suns that never set.

The Arabian oryx were brought back from extinction, bred in captivity, released to protected reserves. They move through the dunes as they always have, dust rising from their hooves, while overhead planes descend toward the airport, carrying people who will never see them.'''
}

def set_memories():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for mind_key, drift_text in DUBAI_MEMORIES.items():
        # Get mind_id
        result = cursor.execute(
            "SELECT mind_id FROM minds WHERE mind_key = ?",
            (mind_key,)
        ).fetchone()

        if not result:
            print(f"⚠️  Mind not found: {mind_key}")
            continue

        mind_id = result[0]

        # Update or insert drift text for version 0
        cursor.execute("""
            UPDATE drift_memory
            SET drift_text = ?
            WHERE mind_id = ? AND version = 0
        """, (drift_text, mind_id))

        if cursor.rowcount == 0:
            # Insert if doesn't exist
            cursor.execute("""
                INSERT INTO drift_memory (mind_id, tick_id, version, drift_text, summary_text, params_json)
                VALUES (?, 0, 0, ?, ?, '{}')
            """, (mind_id, drift_text, f'{mind_key.capitalize()} remembers Dubai'))

        print(f"✅ Set Dubai memory for: {mind_key}")

    conn.commit()
    conn.close()
    print("\n✅ All Dubai memories set for original temporalities!")

if __name__ == '__main__':
    set_memories()
