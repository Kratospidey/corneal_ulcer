from __future__ import annotations

from pathlib import Path
from typing import Any


def get_console():
    try:
        from rich.console import Console  # type: ignore

        return Console(highlight=False, soft_wrap=True)
    except Exception:
        return None


def _print(console, message: str) -> None:
    if console is None:
        print(message)
        return
    console.print(message)


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _metric_style(value: Any, *, reverse: bool = False) -> str:
    if value is None:
        return "white"
    numeric = float(value)
    if reverse:
        if numeric <= 0.05:
            return "green"
        if numeric <= 0.10:
            return "cyan"
        if numeric <= 0.20:
            return "yellow"
        return "red"
    if numeric >= 0.85:
        return "green"
    if numeric >= 0.75:
        return "cyan"
    if numeric >= 0.60:
        return "yellow"
    return "red"


def _render_panel(console, title: str, body: str, style: str = "cyan") -> None:
    if console is None:
        print(f"{title}\n{body}")
        return
    from rich.panel import Panel  # type: ignore

    console.print(Panel.fit(body, title=title, border_style=style))


def emit_run_header(
    console,
    *,
    title: str,
    experiment_name: str,
    task_name: str,
    device: str,
    config_path: str,
    output_root: str | Path,
) -> None:
    body = (
        f"[bold]Experiment[/bold]: {experiment_name}\n"
        f"[bold]Task[/bold]: {task_name}\n"
        f"[bold]Device[/bold]: {device}\n"
        f"[bold]Config[/bold]: {config_path}\n"
        f"[bold]Outputs[/bold]: {output_root}"
    )
    _render_panel(console, title, body)


def emit_dataset_summary(
    console,
    *,
    datasets: dict[str, Any],
    class_names: list[str] | tuple[str, ...],
    batch_size: int,
    epochs: int,
    preprocessing_mode: str,
    train_transform_profile: str,
    sampler_name: str,
) -> None:
    if console is None:
        train_counts = datasets["train"].class_counts()
        _print(
            console,
            "Dataset summary: "
            f"train={len(datasets['train'])} val={len(datasets['val'])} test={len(datasets['test'])} "
            f"batch={batch_size} epochs={epochs} input={preprocessing_mode} aug={train_transform_profile} "
            f"sampler={sampler_name} class_counts={train_counts}",
        )
        return

    from rich.table import Table  # type: ignore

    split_table = Table(title="Dataset Splits", box=None, show_header=True, header_style="bold magenta")
    split_table.add_column("Split", style="bold")
    split_table.add_column("Samples", justify="right")
    split_table.add_column("Class Counts")
    for split_name in ("train", "val", "test"):
        counts = datasets[split_name].class_counts()
        counts_text = ", ".join(f"{class_name}={counts.get(class_name, 0)}" for class_name in class_names)
        split_table.add_row(split_name, str(len(datasets[split_name])), counts_text)
    console.print(split_table)

    recipe_table = Table(title="Run Recipe", box=None, show_header=False)
    recipe_table.add_column("Key", style="bold cyan")
    recipe_table.add_column("Value")
    recipe_table.add_row("Batch size", str(batch_size))
    recipe_table.add_row("Epoch budget", str(epochs))
    recipe_table.add_row("Preprocessing", preprocessing_mode)
    recipe_table.add_row("Train aug", train_transform_profile)
    recipe_table.add_row("Sampler", sampler_name)
    console.print(recipe_table)


def emit_model_summary(console, *, model, training_config: dict[str, Any], class_weights=None) -> None:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if console is None:
        _print(
            console,
            "Model summary: "
            f"name={training_config['model']['name']} total_params={total_params:,} "
            f"trainable_params={trainable_params:,} loss={training_config.get('loss_name', 'weighted_ce')} "
            f"lr={training_config.get('lr', '-')}",
        )
        return

    from rich.table import Table  # type: ignore

    table = Table(title="Model and Optimization", box=None, show_header=False)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")
    table.add_row("Backbone", str(training_config["model"]["name"]))
    table.add_row("Total params", f"{total_params:,}")
    table.add_row("Trainable params", f"{trainable_params:,}")
    table.add_row("Loss", str(training_config.get("loss_name", "weighted_ce")))
    table.add_row("Label smoothing", _format_float(training_config.get("label_smoothing", 0.0), digits=3))
    table.add_row("Optimizer", str(training_config.get("optimizer", "adamw")))
    table.add_row("Learning rate", _format_float(training_config.get("lr"), digits=6))
    table.add_row("Weight decay", _format_float(training_config.get("weight_decay"), digits=6))
    table.add_row("Scheduler", str(training_config.get("scheduler", "cosine")))
    table.add_row("AMP", str(bool(training_config.get("amp", True))))
    if class_weights is not None:
        weights = [f"{float(value):.3f}" for value in class_weights.detach().cpu().tolist()]
        table.add_row("Class weights", ", ".join(weights))
    console.print(table)


def emit_epoch_summary(
    console,
    *,
    epoch: int,
    epochs: int,
    train_loss: float,
    val_loss: float | None,
    val_balanced_accuracy: float,
    val_macro_f1: float,
    best_metric_name: str,
    best_metric: float,
    best_epoch: int,
    improved: bool,
) -> None:
    status = "improved" if improved else "tracked"
    style = "green" if improved else "cyan"
    message = (
        f"[bold]{status}[/bold] "
        f"epoch {epoch}/{epochs} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={_format_float(val_loss)} | "
        f"val_bal_acc={val_balanced_accuracy:.4f} | "
        f"val_macro_f1={val_macro_f1:.4f} | "
        f"best_{best_metric_name}={best_metric:.4f} @ epoch {best_epoch}"
    )
    if console is None:
        print(message.replace("[bold]", "").replace("[/bold]", ""))
        return
    console.print(message, style=style)


def emit_split_metrics(
    console,
    *,
    split_name: str,
    metrics: dict[str, Any],
    class_names: list[str] | tuple[str, ...],
    report_path: str | Path | None = None,
) -> None:
    classification_report = metrics.get("classification_report", {})
    if console is None:
        primary_line = (
            f"{split_name} metrics: "
            f"acc={_format_float(metrics.get('accuracy'))} "
            f"bal_acc={_format_float(metrics.get('balanced_accuracy'))} "
            f"macro_f1={_format_float(metrics.get('macro_f1'))} "
            f"weighted_f1={_format_float(metrics.get('weighted_f1'))} "
            f"ece={_format_float(metrics.get('ece'))}"
        )
        _print(console, primary_line)
        return

    from rich.table import Table  # type: ignore

    primary = Table(title=f"{split_name.title()} Metrics", box=None, show_header=False)
    primary.add_column("Metric", style="bold cyan")
    primary.add_column("Value")
    for metric_name, reverse in (
        ("accuracy", False),
        ("balanced_accuracy", False),
        ("macro_f1", False),
        ("weighted_f1", False),
        ("roc_auc_macro_ovr", False),
        ("pr_auc_macro_ovr", False),
        ("ece", True),
        ("loss", True),
    ):
        value = metrics.get(metric_name)
        style = _metric_style(value, reverse=reverse)
        primary.add_row(metric_name, f"[{style}]{_format_float(value)}[/{style}]")
    if report_path is not None:
        primary.add_row("report", str(report_path))
    console.print(primary)

    per_class = Table(title=f"{split_name.title()} Per-Class Scores", box=None, header_style="bold magenta")
    per_class.add_column("Class", style="bold")
    per_class.add_column("Precision", justify="right")
    per_class.add_column("Recall", justify="right")
    per_class.add_column("F1", justify="right")
    per_class.add_column("Support", justify="right")
    worst_class = None
    worst_recall = float("inf")
    for class_name in class_names:
        payload = classification_report.get(class_name, {})
        recall = float(payload.get("recall", 0.0))
        if recall < worst_recall:
            worst_recall = recall
            worst_class = class_name
        per_class.add_row(
            class_name,
            f"[{_metric_style(payload.get('precision'))}]{_format_float(payload.get('precision'))}[/{_metric_style(payload.get('precision'))}]",
            f"[{_metric_style(recall)}]{_format_float(recall)}[/{_metric_style(recall)}]",
            f"[{_metric_style(payload.get('f1-score'))}]{_format_float(payload.get('f1-score'))}[/{_metric_style(payload.get('f1-score'))}]",
            str(int(payload.get("support", 0))),
        )
    console.print(per_class)
    if worst_class is not None:
        console.print(f"[yellow]Worst recall:[/yellow] {worst_class} ({worst_recall:.4f})")


def emit_figure_summary(console, *, figure_output_dir: str | Path, figure_manifest_path: str | Path | None = None) -> None:
    body = f"[bold]Figure bundle[/bold]: {figure_output_dir}"
    if figure_manifest_path is not None:
        body += f"\n[bold]Manifest[/bold]: {figure_manifest_path}"
    _render_panel(console, "Saved Figures", body, style="green")


def emit_artifact_summary(
    console,
    *,
    checkpoint_path: str | Path,
    exported_checkpoint_path: str | Path | None,
    metrics_dir: str | Path,
    reports_dir: str | Path,
) -> None:
    body = f"[bold]Checkpoint[/bold]: {checkpoint_path}\n[bold]Metrics[/bold]: {metrics_dir}\n[bold]Reports[/bold]: {reports_dir}"
    if exported_checkpoint_path is not None:
        body += f"\n[bold]Exported[/bold]: {exported_checkpoint_path}"
    _render_panel(console, "Artifacts", body, style="blue")
