from __future__ import annotations


def build_sampler(dataset, sampler_name: str):
    import torch  # type: ignore
    from torch.utils.data import WeightedRandomSampler  # type: ignore

    name = sampler_name.lower()
    if name in {"none", "off"}:
        return None
    if name not in {"weighted", "weighted_random"}:
        raise ValueError(f"Unsupported sampler: {sampler_name}")

    class_counts = dataset.class_counts()
    weights = []
    for index in range(len(dataset)):
        sample = dataset.rows.iloc[index]
        label = str(sample[dataset.label_column])
        weights.append(1.0 / max(1, class_counts[label]))
    weights_tensor = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights_tensor, num_samples=len(weights), replacement=True)
