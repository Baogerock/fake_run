import argparse
import os
from typing import List

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS: List[str] = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fake labels for images.")
    parser.add_argument(
        "images",
        type=str,
        help="Path to the folder containing source images."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_dir = os.path.abspath(args.images)
    label_dir = f"{image_dir}_labels"

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    os.makedirs(label_dir, exist_ok=True)

    image_files = sorted(
        [
            os.path.join(image_dir, name)
            for name in os.listdir(image_dir)
            if os.path.splitext(name.lower())[1] in IMAGE_EXTENSIONS
        ]
    )

    if not image_files:
        raise RuntimeError(f"No supported image files found in {image_dir}.")

    for image_path in image_files:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_dir, f"{base_name}.npy")

        with Image.open(image_path) as image:
            width, height = image.size

        fake_mask = np.random.rand(height, width).astype(np.float32)
        np.save(label_path, fake_mask)
        print(f"Saved fake label: {label_path}")

    print(f"Generated {len(image_files)} fake labels in {label_dir}.")


if __name__ == "__main__":
    main()
