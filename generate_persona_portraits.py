#!/usr/bin/env python3
"""
Generate three persona portrait images for Athens temporalities.
Creates portrait-style images (upper body, above shoulder line to head)
of local Athens residents with temporality-specific backgrounds.
"""

import os
import sys
from pathlib import Path
from openai import OpenAI

# Constants
OUTPUT_DIR = Path(__file__).parent / "data" / "persona_portraits"
OUTPUT_DIR.mkdir(exist_ok=True)

# Portrait specifications for each temporality
PORTRAITS = {
    "human": {
        "persona": (
            "A portrait photograph of an Athens resident in their 30s-40s, "
            "captured from shoulders to head. Natural, contemplative expression. "
            "Person is sitting in a Kolonaki café interior. Background shows: "
            "warm café interior, geometric shadows from awnings on marble surfaces, "
            "soft afternoon light filtering through windows, blurred café patrons. "
            "Color palette: warm amber, golden café tones, soft browns. "
            "Intimate, personal atmosphere."
        ),
        "mood": (
            "Quiet observation, slight melancholy, urban sophistication. "
            "Someone embedded in Athens social life but feeling a subtle distance. "
            "Natural lighting, soft focus background, person in sharp focus."
        ),
    },
    "environment": {
        "persona": (
            "A portrait photograph of an Athens resident in their 50s-60s, "
            "captured from shoulders to head. Weathered, contemplative expression. "
            "Person is standing in the National Garden in winter. Background shows: "
            "misty garden paths, pine trees, eucalyptus branches, winter atmosphere, "
            "soft natural light through tree canopy, blurred garden visitors. "
            "Color palette: muted greens, grays, winter earth tones. "
            "Elemental, connected to natural cycles."
        ),
        "mood": (
            "Quiet wisdom, attunement to weather and seasons, patience. "
            "Someone who observes natural forces shaping the city. "
            "Natural winter light, soft focus background, person in focus."
        ),
    },
    "digital": {
        "persona": (
            "A portrait photograph of an Athens resident in their 20s-30s, "
            "captured from shoulders to head. Alert, analytical expression. "
            "Person is standing on an Exarchia street at night. Background shows: "
            "ATM screens glowing blue-white, LED lights, graffiti-covered walls, "
            "digital parking meters, phone screen light illuminating face, "
            "blurred urban technology layer. "
            "Color palette: blue-white LED glow, deep teal, cool urban tones. "
            "Technological, contested urban space."
        ),
        "mood": (
            "Sharp awareness, digital consciousness, urban resistance. "
            "Someone documenting the friction between technology and politics. "
            "LED backlight, screen glow on face, shallow depth of field."
        ),
    },
}


def generate_persona_portrait(
    temporality: str,
    api_key: str,
    style_reference: str = "",
) -> str:
    """Generate a persona portrait for a specific temporality."""

    persona_spec = PORTRAITS[temporality]

    # Construct the prompt
    prompt = f"""
A portrait photograph showing the upper body (shoulders to head) of a person.

SUBJECT:
{persona_spec['persona']}

MOOD & ATMOSPHERE:
{persona_spec['mood']}

VISUAL STYLE:
Photographic portrait with artistic, moody aesthetic. NOT a literal documentary photo.
Soft, dreamlike quality with gentle focus on the person's face and upper body.
Background slightly blurred (shallow depth of field) to emphasize the subject.
Natural, authentic expression - not posed or artificial.
Cinematic color grading, atmospheric lighting.
Clean image with smooth gradations, NO grain, NO noise, NO digital artifacts.

The portrait should feel intimate and real - like a character study that captures
both the person and the urban environment they inhabit.

REFERENCE STYLE:
{style_reference}

IMPORTANT CONSTRAINTS:
- Portrait orientation (2:3 ratio)
- Person should represent diverse Athens demographics (not stereotypical)
- Natural, authentic appearance
- No exaggerated features or caricature
- Background shows the temporality's context but doesn't overpower the subject
- Person is in focus, background has subtle blur
- Professional portrait photography aesthetic
"""

    print(f"\n📸 Generating {temporality.upper()} persona portrait...")
    print(f"   Style: {persona_spec['persona'][:80]}...")

    client = OpenAI(api_key=api_key)

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1792",  # Portrait orientation
            quality="hd",
            n=1,
        )

        image_url = response.data[0].url
        print(f"   ✓ Image generated: {image_url[:60]}...")

        # Download the image
        import urllib.request
        output_path = OUTPUT_DIR / f"{temporality}_persona.png"

        with urllib.request.urlopen(image_url) as url_response:
            image_data = url_response.read()

        with open(output_path, 'wb') as f:
            f.write(image_data)

        print(f"   ✓ Saved to: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"   ✗ Error: {e}")
        raise


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate persona portraits for Athens temporalities"
    )
    parser.add_argument(
        "--style-reference",
        type=str,
        default="Artistic portrait photography with moody, cinematic aesthetic. Natural lighting.",
        help="Style reference description from uploaded images"
    )
    parser.add_argument(
        "--temporalities",
        type=str,
        nargs="+",
        default=["human", "environment", "digital"],
        help="Which temporalities to generate (default: all three)"
    )

    args = parser.parse_args()

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🎭 KEEPSAKE DRIFT PERSONA PORTRAIT GENERATOR")
    print("=" * 70)
    print(f"\nGenerating {len(args.temporalities)} portrait(s)...")
    print(f"Style reference: {args.style_reference[:60]}...")
    print(f"Output directory: {OUTPUT_DIR}")

    generated = []

    for temporality in args.temporalities:
        if temporality not in PORTRAITS:
            print(f"\n⚠️  Unknown temporality: {temporality}")
            print(f"   Available: {list(PORTRAITS.keys())}")
            continue

        try:
            output_path = generate_persona_portrait(
                temporality,
                api_key,
                args.style_reference
            )
            generated.append(output_path)

        except Exception as e:
            print(f"\n❌ Failed to generate {temporality} portrait: {e}")
            continue

    print("\n" + "=" * 70)
    print(f"✅ Generated {len(generated)}/{len(args.temporalities)} portraits")
    print("=" * 70)

    if generated:
        print("\n📁 Output files:")
        for path in generated:
            print(f"   - {path}")

    print("\n💡 Next step: Use these portraits with create_museum_mockup.py")
    print()


if __name__ == "__main__":
    main()
