import argparse
import os
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from efficientunet.efficientunet import get_efficientunet_b0


IMAGE_EXTENSIONS: List[str] = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
]


class ImageMaskDataset(Dataset):
    """Dataset that pairs images with fake labels stored as NumPy arrays."""

    def __init__(self, image_dir: str):
        self.image_dir = os.path.abspath(image_dir)
        self.label_dir = f"{self.image_dir}_labels"

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(
                f"Label directory not found: {self.label_dir}. "
                "Run make_label.py to generate fake labels."
            )

        self.image_paths = sorted(
            [
                os.path.join(self.image_dir, name)
                for name in os.listdir(self.image_dir)
                if os.path.splitext(name.lower())[1] in IMAGE_EXTENSIONS
            ]
        )

        if not self.image_paths:
            raise RuntimeError(f"No supported image files found in {self.image_dir}.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(self.label_dir, f"{base_name}.npy")

        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Missing label file: {label_path}")

        image = Image.open(image_path).convert("RGB")
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        mask = np.load(label_path).astype(np.float32)
        if mask.ndim == 2:
            mask = mask[np.newaxis, ...]
        mask_tensor = torch.from_numpy(mask)

        return image_tensor, mask_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientUNet on fake labels.")
    parser.add_argument(
        "images",
        type=str,
        help="Path to the folder containing training images."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = ImageMaskDataset(args.images)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=False)
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 2
    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, (images, masks) in enumerate(data_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Step {step}/{len(data_loader)} "
                f"Loss: {loss.item():.4f}"
            )

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")

    print("Training finished. No model files were saved.")


if __name__ == "__main__":
    main()
