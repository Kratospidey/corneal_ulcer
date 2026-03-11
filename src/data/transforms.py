from __future__ import annotations


def build_train_transform(image_size: int = 224):
    from torchvision import transforms  # type: ignore

    return transforms.Compose(
        [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_eval_transform(image_size: int = 224):
    from torchvision import transforms  # type: ignore

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_transforms(image_size: int = 224) -> dict[str, object]:
    return {
        "train": build_train_transform(image_size=image_size),
        "val": build_eval_transform(image_size=image_size),
        "test": build_eval_transform(image_size=image_size),
    }
