from typing import Tuple
from PIL import Image
import torch
from torchvision import transforms  # type: ignore

class TwoViewTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base_transform(x), self.base_transform(x)

def build_ssl_train_transforms(image_size: int = 224):
    """
    Builds slit-lamp-safe augmentations for SSL pretraining.
    Returns a TwoViewTransform that produces two different augmented views.
    """
    base_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return TwoViewTransform(base_transform)

def build_ssl_eval_transforms(image_size: int = 224):
    """
    Deterministic transform for SSL validation / sanity checks.
    """
    return transforms.Compose([
        transforms.Resize(int(image_size / 0.875)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
