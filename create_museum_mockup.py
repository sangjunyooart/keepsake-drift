#!/usr/bin/env python3
"""
Museum Installation Mockup Generator for Keepsake Drift
Creates a three-screen triptych showing different temporality personas
with bilingual subtitles and QR codes.
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Tuple

import qrcode
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

# Constants
DB_PATH = Path(__file__).parent / "data" / "keepsake.sqlite"
IMAGE_DIR = Path(__file__).parent / "data" / "images"
OUTPUT_DIR = Path(__file__).parent

# Layout specifications (adjustable based on reference images)
SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 1920
SCREEN_GAP = 30
PORTRAIT_HEIGHT = 1400
SUBTITLE_AREA_HEIGHT = 520

# Typography
ENGLISH_FONT_SIZE = 40
GREEK_FONT_SIZE = 36
LINE_SPACING = 1.3
TEXT_COLOR = (255, 255, 255)  # White
BG_COLOR = (26, 26, 26)  # Museum dark gray

# QR Code
QR_SIZE = 200
QR_ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_H

# Temporalities to display
TEMPORALITIES = ["human", "environment", "digital"]


def query_drift_text(db_path: Path, mind_key: str, version: int) -> Tuple[str, str]:
    """Query English and Greek drift text for a specific temporality."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    SELECT d.drift_text, d.drift_text_ar
    FROM drift_memory d
    JOIN minds m ON d.mind_id = m.mind_id
    WHERE m.mind_key = ? AND d.version = ?
    """

    cursor.execute(query, (mind_key, version))
    result = cursor.fetchone()
    conn.close()

    if not result:
        raise ValueError(f"No drift found for {mind_key} at version {version}")

    drift_text_en, drift_text_gr = result

    # Truncate to ~200 characters for subtitle display
    if len(drift_text_en) > 200:
        drift_text_en = drift_text_en[:197] + "..."
    if len(drift_text_gr) > 200:
        drift_text_gr = drift_text_gr[:197] + "..."

    return drift_text_en, drift_text_gr


def generate_portrait_image(
    mind_key: str,
    drift_text: str,
    api_key: str,
    output_path: Optional[Path] = None
) -> Image.Image:
    """Generate atmospheric portrait image using OpenAI."""

    # Portrait-specific visual styles per temporality
    visual_styles = {
        "human": (
            "A portrait-oriented atmospheric image suggesting a human presence in an intimate interior. "
            "NOT photorealistic, NOT a literal face. Soft silhouette seen through a café window, "
            "geometric shadows cast across warm surfaces. Doorway framing a blurred figure. "
            "Color palette: warm amber, golden tones, soft browns. Intimate scale, close framing."
        ),
        "environment": (
            "A portrait-oriented atmospheric image suggesting an elemental presence in nature. "
            "NOT photorealistic, NOT a literal face. Blurred walking figure among garden paths, "
            "weather-worn textures, natural patterns. Mist and foliage creating abstract shapes. "
            "Color palette: muted greens, grays, winter atmosphere. Horizontal composition with weather visible."
        ),
        "digital": (
            "A portrait-oriented atmospheric image suggesting a technological presence. "
            "NOT photorealistic, NOT a literal face. Screen glow reflecting on indistinct forms, "
            "LED light casting blue-white illumination on silhouettes. Abstract digital patterns. "
            "Color palette: blue-white LED glow, deep teal, technological atmosphere. Urban tech layer."
        ),
    }

    style = visual_styles.get(mind_key, visual_styles["human"])

    prompt = f"""
    VISUAL STYLE (mandatory):
    {style}

    Softly focused with gentle, dreamlike clarity — NOT sharp, but NOT grainy or noisy.
    Clean image with smooth gradations. Subtle softness, like shallow depth of field.
    Edges are gentle, not razor-sharp. Rich dark shadows with soft transitions.
    NO grain, NO noise, NO film texture, NO digital artifacts.

    The mood is quiet, melancholic, intimate — like a fading memory that retains
    its emotional weight but has lost its crispness. A photograph that feels more
    like remembering than seeing.

    EMOTIONAL CONTEXT:
    {drift_text[:300]}

    Portrait orientation (2:3). No text, no words, no letters, no writing.
    No recognizable human faces (blur or silhouette only).
    No culturally or religiously offensive imagery.
    """

    print(f"  Generating portrait for {mind_key}...")

    client = OpenAI(api_key=api_key)

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1792",  # Portrait orientation
        quality="hd",
        n=1,
    )

    image_url = response.data[0].url

    # Download the image
    import urllib.request
    import io

    with urllib.request.urlopen(image_url) as url_response:
        image_data = url_response.read()

    image = Image.open(io.BytesIO(image_data))

    if output_path:
        image.save(output_path)
        print(f"  ✓ Saved portrait to {output_path}")

    return image


def generate_qr_code(url: str, size: int = QR_SIZE) -> Image.Image:
    """Generate QR code as PIL Image."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=QR_ERROR_CORRECTION,
        box_size=10,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)

    qr_image = qr.make_image(fill_color="black", back_color="white")
    qr_image = qr_image.resize((size, size), Image.Resampling.LANCZOS)

    return qr_image


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Get a font with proper Unicode support for Greek."""
    # Try to find system fonts with Greek support
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue

    # Fallback to default font
    return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Wrap text to fit within max_width."""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = font.getbbox(test_line)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines


def create_subtitle_panel(
    english_text: str,
    greek_text: str,
    qr_image: Image.Image,
    width: int,
    height: int
) -> Image.Image:
    """Create subtitle panel with bilingual text and QR code."""
    panel = Image.new('RGB', (width, height), BG_COLOR)
    draw = ImageDraw.Draw(panel)

    # Load fonts
    english_font = get_font(ENGLISH_FONT_SIZE, bold=True)
    greek_font = get_font(GREEK_FONT_SIZE, bold=False)

    # Text margin
    text_margin = 60
    max_text_width = width - (text_margin * 2)

    # Wrap text
    english_lines = wrap_text(english_text, english_font, max_text_width)
    greek_lines = wrap_text(greek_text, greek_font, max_text_width)

    # Calculate vertical positioning
    line_height_en = int(ENGLISH_FONT_SIZE * LINE_SPACING)
    line_height_gr = int(GREEK_FONT_SIZE * LINE_SPACING)

    english_block_height = len(english_lines) * line_height_en
    greek_block_height = len(greek_lines) * line_height_gr
    gap_between_languages = 30
    qr_top_margin = 40

    total_content_height = (
        english_block_height +
        gap_between_languages +
        greek_block_height +
        qr_top_margin +
        QR_SIZE
    )

    # Start y position (centered vertically)
    start_y = (height - total_content_height) // 2

    # Draw English text (center-aligned)
    y = start_y
    for line in english_lines:
        bbox = english_font.getbbox(line)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, y), line, font=english_font, fill=TEXT_COLOR)
        y += line_height_en

    # Draw Greek text (center-aligned)
    y += gap_between_languages
    for line in greek_lines:
        bbox = greek_font.getbbox(line)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, y), line, font=greek_font, fill=TEXT_COLOR)
        y += line_height_gr

    # Paste QR code (centered)
    y += qr_top_margin
    qr_x = (width - QR_SIZE) // 2
    panel.paste(qr_image, (qr_x, y))

    return panel


def composite_three_screen_mockup(
    portraits: list[Image.Image],
    subtitles: list[Image.Image],
    output_path: Path
) -> Image.Image:
    """Composite three screens side-by-side into final mockup."""

    # Calculate total dimensions
    total_width = (SCREEN_WIDTH * 3) + (SCREEN_GAP * 2)
    total_height = SCREEN_HEIGHT

    # Create canvas
    canvas = Image.new('RGB', (total_width, total_height), BG_COLOR)

    # Composite each screen
    for i in range(3):
        x_offset = i * (SCREEN_WIDTH + SCREEN_GAP)

        # Resize portrait to fit
        portrait = portraits[i].resize(
            (SCREEN_WIDTH, PORTRAIT_HEIGHT),
            Image.Resampling.LANCZOS
        )

        # Paste portrait
        canvas.paste(portrait, (x_offset, 0))

        # Paste subtitle panel
        canvas.paste(subtitles[i], (x_offset, PORTRAIT_HEIGHT))

    # Save
    canvas.save(output_path, quality=95)
    print(f"\n✓ Mockup saved to: {output_path}")
    print(f"  Dimensions: {total_width}x{total_height}px")

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Generate museum installation mockup for Keepsake Drift"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=31,
        help="Drift version to use (default: 31)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="museum_installation_mockup.png",
        help="Output filename (default: museum_installation_mockup.png)"
    )
    parser.add_argument(
        "--qr-base-url",
        type=str,
        default="https://example.com/keepsake-drift",
        help="Base URL for QR codes (default: https://example.com/keepsake-drift)"
    )
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Use existing generated images instead of creating new ones"
    )
    parser.add_argument(
        "--generate-portraits",
        action="store_true",
        help="Generate new portrait-style images (requires OpenAI API key)"
    )

    args = parser.parse_args()

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if args.generate_portraits and not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Either set it or use --use-existing flag")
        sys.exit(1)

    print(f"\n🖼️  Keepsake Drift Museum Installation Mockup Generator")
    print(f"=" * 60)
    print(f"Version: {args.version}")
    print(f"Output: {args.output}")
    print(f"QR Base URL: {args.qr_base_url}")
    print(f"Mode: {'Generate new portraits' if args.generate_portraits else 'Use existing images'}")
    print()

    portraits = []
    subtitles = []

    for mind_key in TEMPORALITIES:
        print(f"\n📍 Processing {mind_key.upper()}...")

        # Query drift texts
        print(f"  Querying database...")
        try:
            drift_en, drift_gr = query_drift_text(DB_PATH, mind_key, args.version)
            print(f"  ✓ English: {drift_en[:50]}...")
            print(f"  ✓ Greek: {drift_gr[:50]}...")
        except ValueError as e:
            print(f"  ✗ Error: {e}")
            sys.exit(1)

        # Get or generate portrait image
        if args.generate_portraits:
            portrait = generate_portrait_image(
                mind_key,
                drift_en,
                api_key,
                output_path=IMAGE_DIR / f"{mind_key}_mockup.png"
            )
        else:
            # Use existing image
            existing_path = IMAGE_DIR / f"{mind_key}_v{args.version}.png"
            if not existing_path.exists():
                # Try latest available version
                existing_images = sorted(IMAGE_DIR.glob(f"{mind_key}_v*.png"))
                if not existing_images:
                    print(f"  ✗ No existing images found for {mind_key}")
                    sys.exit(1)
                existing_path = existing_images[-1]
                print(f"  Using {existing_path.name}")

            portrait = Image.open(existing_path)
            print(f"  ✓ Loaded existing image: {existing_path.name}")

        portraits.append(portrait)

        # Generate QR code
        qr_url = f"{args.qr_base_url}/{mind_key}"
        print(f"  Generating QR code for: {qr_url}")
        qr_image = generate_qr_code(qr_url)
        print(f"  ✓ QR code generated")

        # Create subtitle panel
        print(f"  Creating subtitle panel...")
        subtitle_panel = create_subtitle_panel(
            drift_en,
            drift_gr,
            qr_image,
            SCREEN_WIDTH,
            SUBTITLE_AREA_HEIGHT
        )
        subtitles.append(subtitle_panel)
        print(f"  ✓ Subtitle panel created")

    # Composite final mockup
    print(f"\n🎨 Compositing final three-screen mockup...")
    output_path = OUTPUT_DIR / args.output
    composite_three_screen_mockup(portraits, subtitles, output_path)

    print(f"\n✅ Done! Museum installation mockup created successfully.")
    print(f"\n📁 Output file: {output_path}")
    print(f"📏 Dimensions: {(SCREEN_WIDTH * 3) + (SCREEN_GAP * 2)}x{SCREEN_HEIGHT}px")
    print(f"\n💡 Tip: Scan the QR codes to verify they work correctly.\n")


if __name__ == "__main__":
    main()
