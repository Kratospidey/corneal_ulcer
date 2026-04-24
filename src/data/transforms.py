from __future__ import annotations


def build_train_transform(image_size: int = 224, train_profile: str = "default"):
    from torchvision import transforms  # type: ignore

    profile = str(train_profile).lower()
    if profile == "pattern_augplus_v2":
        return transforms.Compose(
            [
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0), ratio=(0.94, 1.06)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=14,
                    translate=(0.04, 0.04),
                    scale=(0.94, 1.06),
                    fill=0,
                ),
                transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.12, hue=0.018),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.75))], p=0.2),
                transforms.RandomAdjustSharpness(sharpness_factor=1.15, p=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

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


def build_transforms(image_size: int = 224, train_profile: str = "default") -> dict[str, object]:
    return {
        "train": build_train_transform(image_size=image_size, train_profile=train_profile),
        "val": build_eval_transform(image_size=image_size),
        "test": build_eval_transform(image_size=image_size),
    }
