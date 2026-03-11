from __future__ import annotations

from pathlib import Path


def disable_inplace_relu(module) -> None:
    import torch  # type: ignore

    for child in module.modules():
        if isinstance(child, torch.nn.ReLU):
            child.inplace = False


class GradCAM:
    def __init__(self, model, target_layer) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):  # noqa: ARG002
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):  # noqa: ARG002
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, target_index: int):
        import torch  # type: ignore

        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        score = logits[:, target_index].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp(min=1e-8)
        return cam.cpu().numpy()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def overlay_cam_on_image(image, cam_array, alpha: float = 0.35):
    import matplotlib.cm as cm  # type: ignore
    import numpy as np  # type: ignore
    from PIL import Image  # type: ignore

    base = image.convert("RGB")
    heatmap = cm.get_cmap("jet")(cam_array)[:, :, :3]
    heatmap = (heatmap * 255).astype("uint8")
    heatmap_image = Image.fromarray(heatmap).resize(base.size)
    return Image.blend(base, heatmap_image, alpha=alpha)


def save_overlay(image, cam_array, output_path: str | Path, title: str = "") -> None:
    from PIL import ImageDraw  # type: ignore

    overlay = overlay_cam_on_image(image, cam_array)
    draw = ImageDraw.Draw(overlay)
    if title:
        draw.text((8, 8), title, fill=(255, 255, 255))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
