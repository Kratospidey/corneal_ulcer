from __future__ import annotations


def build_sampler(dataset, sampler_name: str, sampler_temperature: float = 1.0):
    import torch  # type: ignore
    from torch.utils.data import WeightedRandomSampler  # type: ignore

    name = sampler_name.lower()
    if name in {"none", "off"}:
        return None
    if name not in {"weighted", "weighted_random", "weighted_sampler_tempered"}:
        raise ValueError(f"Unsupported sampler: {sampler_name}")

    class_counts = dataset.class_counts()
    weights = []
    temperature = float(max(sampler_temperature, 1e-6))
    for index in range(len(dataset)):
        sample = dataset.rows.iloc[index]
        label = str(sample[dataset.label_column])
        base_weight = 1.0 / max(1, class_counts[label])
        if name == "weighted_sampler_tempered":
            base_weight = base_weight**temperature
        weights.append(base_weight)
    weights_tensor = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights_tensor, num_samples=len(weights), replacement=True)
