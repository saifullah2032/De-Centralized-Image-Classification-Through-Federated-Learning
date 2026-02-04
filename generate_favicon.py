#!/usr/bin/env python3
"""
Generate PNG favicons from SVG
This is a simple fallback generator
"""

from PIL import Image, ImageDraw


# Create a simple favicon (32x32 and 180x180 for Apple)
def create_favicon(size, filename):
    # Create image with transparent background
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw background circle (primary color)
    margin = 2
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=(99, 102, 241, 255),
        outline=(79, 70, 229, 255),
    )

    # Draw network nodes (simplified)
    node_radius = size // 15

    # Center top node
    draw.ellipse(
        [
            size // 2 - node_radius,
            size // 3 - node_radius,
            size // 2 + node_radius,
            size // 3 + node_radius,
        ],
        fill=(255, 255, 255, 255),
    )

    # Left node
    draw.ellipse(
        [
            size // 3 - node_radius,
            size // 2 - node_radius,
            size // 3 + node_radius,
            size // 2 + node_radius,
        ],
        fill=(255, 255, 255, 255),
    )

    # Right node
    draw.ellipse(
        [
            2 * size // 3 - node_radius,
            size // 2 - node_radius,
            2 * size // 3 + node_radius,
            size // 2 + node_radius,
        ],
        fill=(255, 255, 255, 255),
    )

    # Bottom center node (larger)
    large_radius = node_radius + 2
    draw.ellipse(
        [
            size // 2 - large_radius,
            2 * size // 3 - large_radius,
            size // 2 + large_radius,
            2 * size // 3 + large_radius,
        ],
        fill=(255, 255, 255, 255),
    )

    # Save
    img.save(filename, "PNG")
    print(f"[OK] Created {filename} ({size}x{size})")


if __name__ == "__main__":
    import os

    # Get static directory
    static_dir = "frontend_web/static"

    if not os.path.exists(static_dir):
        print(f"Error: {static_dir} not found")
        exit(1)

    # Generate favicons
    create_favicon(32, os.path.join(static_dir, "favicon.png"))
    create_favicon(180, os.path.join(static_dir, "apple-touch-icon.png"))

    print("\n[SUCCESS] Favicon generation complete!")
