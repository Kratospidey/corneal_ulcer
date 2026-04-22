from __future__ import annotations


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224, augmentation_profile: str = "baseline"):
    from torchvision import transforms  # type: ignore

    profile = augmentation_profile.strip().lower()
    if profile in {"baseline", "default", "standard"}:
        ops = [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
        ]
    elif profile == "augplus_v1":
        ops = [
            transforms.Resize((image_size + 24, image_size + 24)),
            transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=12, translate=(0.03, 0.03), scale=(0.96, 1.04), fill=0),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.1, hue=0.015),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=0.15),
        ]
    elif profile == "augplus_v2":
        ops = [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0), ratio=(0.94, 1.06)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=14, translate=(0.04, 0.04), scale=(0.94, 1.06), fill=0),
            transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.12, hue=0.018),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.75))], p=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=1.15, p=0.15),
        ]
    else:
        raise ValueError(f"Unsupported augmentation_profile: {augmentation_profile}")

    return transforms.Compose(
        [
            *ops,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224):
    from torchvision import transforms  # type: ignore

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_transforms(image_size: int = 224, augmentation_profile: str = "baseline") -> dict[str, object]:
    return {
        "train": build_train_transform(image_size=image_size, augmentation_profile=augmentation_profile),
        "val": build_eval_transform(image_size=image_size),
        "test": build_eval_transform(image_size=image_size),
    }
