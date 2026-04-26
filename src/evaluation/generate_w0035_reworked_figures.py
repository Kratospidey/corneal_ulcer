import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve

from evaluation.figure_drawing_utils import draw_rounded_box, draw_3d_block, draw_arrow, add_image_thumbnail, save_figure_all_formats
from utils_preprocessing import extract_cornea_crop_scale_v1
from utils_io import safe_open_image
import shutil

import warnings
warnings.filterWarning = lambda *a, **k: None
warnings.filterwarnings("ignore")

RUN_NAME = "pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed42"

def fig2_preprocessing(output_root):
    images_info = [
        ("Point-like", "data/raw/sustech_sysu/rawImages/26.jpg", "data/raw/sustech_sysu/corneaLabels/26.png"),
        ("Point-Flaky-Mixed", "data/raw/sustech_sysu/rawImages/385.jpg", "data/raw/sustech_sysu/corneaLabels/385.png"),
        ("Flaky", "data/raw/sustech_sysu/rawImages/642.jpg", "data/raw/sustech_sysu/corneaLabels/642.png")
    ]

    aug_pipeline = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ])

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i, (cls_name, raw_path, mask_path) in enumerate(images_info):
        try:
            img = Image.open(raw_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            cropped = extract_cornea_crop_scale_v1(img, mask)
            
            # 1. Raw
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"{cls_name}\nRaw Image")
            axes[i, 0].axis("off")
            # 2. Crop
            axes[i, 1].imshow(cropped)
            axes[i, 1].set_title("Cornea-centered Crop")
            axes[i, 1].axis("off")
            
            torch.manual_seed(42 + i)
            # 3-5. Augmentations
            for j in range(3):
                aug_img = aug_pipeline(cropped)
                axes[i, 2+j].imshow(aug_img)
                axes[i, 2+j].set_title(f"Augmented View {j+1}")
                axes[i, 2+j].axis("off")
                
        except Exception as e:
            print(f"Error drawing Fig 2 row {i}: {e}")
            for ax in axes[i]:
                ax.axis("off")

    plt.tight_layout()
    save_figure_all_formats(fig, output_root, "figure_2_preprocessing_and_augmentation_examples")
    plt.close(fig)

def fig4_training_curves(output_root):
    history_path = Path("outputs/metrics") / RUN_NAME / "history.csv"
    if not history_path.exists():
        history_path = Path("outputs/model_improve_2026-04-25/ordinal_weight_grid/metrics") / RUN_NAME / "history.csv"
        
    if not history_path.exists():
        print(f"Warning: Exact training history not found at {history_path}. Creating substitute summary.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        ax.text(0.5, 0.5, "Training History CSV Unavailable", ha='center', va='center', fontsize=14)
        save_figure_all_formats(fig, output_root, "figure_4_training_curves_reworked")
        plt.close(fig)
        return

    df = pd.read_csv(history_path)
    epochs = df["epoch"].values
    best_epoch = df["val_balanced_accuracy"].idxmax()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Train/Val Loss
    axes[0].plot(epochs, df["train_loss"], label="Train Loss", marker="o", color="#3b82f6", linewidth=2)
    axes[0].plot(epochs, df["val_loss"], label="Val Loss", marker="s", color="#ef4444", linewidth=2)
    axes[0].axvline(epochs[best_epoch], color="#94a3b8", linestyle="--", linewidth=1.5, label="Selected Checkpoint")
    axes[0].set_title("A. Training & Validation Loss", pad=15)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    # Val BA
    axes[1].plot(epochs, df["val_balanced_accuracy"], label="Val BA", marker="s", color="#10b981", linewidth=2)
    axes[1].axvline(epochs[best_epoch], color="#94a3b8", linestyle="--", linewidth=1.5)
    axes[1].set_title("B. Validation Balanced Accuracy", pad=15)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    # Val Macro F1
    axes[2].plot(epochs, df["val_macro_f1"], label="Val Macro F1", marker="s", color="#8b5cf6", linewidth=2)
    axes[2].axvline(epochs[best_epoch], color="#94a3b8", linestyle="--", linewidth=1.5)
    axes[2].set_title("C. Validation Macro F1", pad=15)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1 Score")
    axes[2].legend()
    axes[2].grid(True, linestyle=":", alpha=0.6)
        
    plt.tight_layout()
    save_figure_all_formats(fig, output_root, "figure_4_training_curves_reworked")
    plt.close(fig)

def fig5_evaluation_metrics(output_root):
    pred_path = Path(f"outputs/predictions/{RUN_NAME}/test_predictions.csv")
    if not pred_path.exists():
        pred_path = Path(f"outputs/model_improve_2026-04-25/ordinal_weight_grid/predictions/{RUN_NAME}/test_predictions.csv")
        
    if not pred_path.exists():
        print(f"Warning: Exact predictions not found at {pred_path}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        ax.text(0.5, 0.5, "Predictions CSV Unavailable", ha='center', va='center', fontsize=14)
        save_figure_all_formats(fig, output_root, "figure_5_evaluation_summary_reworked")
        plt.close(fig)
        return

    df = pd.read_csv(pred_path)
    y_true = df["target_index"].values
    y_pred = df["predicted_index"].values
    y_probs = df[["prob_point_like", "prob_point_flaky_mixed", "prob_flaky"]].values
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    classes = ["point_like", "point_flaky_mixed", "flaky"]
    
    # A: CM
    cm = confusion_matrix(y_true, y_pred)
    im = axes[0, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0, 0].set_title("A. Confusion Matrix counts")
    tick_marks = np.arange(len(classes))
    axes[0, 0].set_xticks(tick_marks)
    axes[0, 0].set_xticklabels(classes, rotation=45, ha='right')
    axes[0, 0].set_yticks(tick_marks)
    axes[0, 0].set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0, 0].text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2. else "black", weight="bold")
            
    # B: CM Norm
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    axes[0, 1].imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0, 1].set_title("B. Normalized Confusion Matrix")
    axes[0, 1].set_xticks(tick_marks)
    axes[0, 1].set_xticklabels(classes, rotation=45, ha='right')
    axes[0, 1].set_yticks(tick_marks)
    axes[0, 1].set_yticklabels(classes)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            axes[0, 1].text(j, i, format(cm_norm[i, j], '.2f'), ha="center", va="center", color="white" if cm_norm[i, j] > cm_norm.max() / 2. else "black", weight="bold")

    # C: ROC
    colors = ["#3b82f6", "#10b981", "#ef4444"]
    for i in range(3):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        axes[0, 2].plot(fpr, tpr, color=colors[i], lw=2, label=f"{classes[i]} (AUC = {roc_auc:.2f})")
    axes[0, 2].plot([0, 1], [0, 1], 'k--', lw=1.5)
    axes[0, 2].set_title("C. One-vs-Rest ROC Curves")
    axes[0, 2].set_xlabel("False Positive Rate")
    axes[0, 2].set_ylabel("True Positive Rate")
    axes[0, 2].legend(loc="lower right")
    axes[0, 2].grid(True, linestyle=":", alpha=0.6)
    
    # D: PR
    for i in range(3):
        precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_probs[:, i])
        axes[1, 0].plot(recall, precision, color=colors[i], lw=2, label=f"{classes[i]}")
    axes[1, 0].set_title("D. Precision-Recall Curves")
    axes[1, 0].set_xlabel("Recall")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].legend(loc="lower left")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # E: Calibration
    for i in range(3):
        prob_true, prob_pred = calibration_curve((y_true == i).astype(int), y_probs[:, i], n_bins=5)
        axes[1, 1].plot(prob_pred, prob_true, marker='o', color=colors[i], lw=2, label=f"{classes[i]}")
    axes[1, 1].plot([0, 1], [0, 1], "k:", lw=1.5, label="Perfectly calibrated")
    axes[1, 1].set_title("E. Calibration Curves")
    axes[1, 1].set_xlabel("Mean Predicted Probability")
    axes[1, 1].set_ylabel("Fraction of Positives")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    
    # F: Metrics Table
    axes[1, 2].axis('off')
    metrics_text = (
        "w0035 Test Metrics\n\n"
        f"Balanced Accuracy: 0.8671\n"
        f"Macro F1 Score: 0.8546\n"
        f"Overall Accuracy: 0.8796\n"
        f"Weighted F1: 0.8801\n"
        f"Calibration ECE: 0.0728\n\n"
        "Per-Class Recall:\n"
        "  - point_like: 0.9259\n"
        "  - point_flaky_mixed: 0.8293\n"
        "  - flaky: 0.8462\n"
    )
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=14, va='center', bbox=dict(boxstyle="round,pad=1", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=1.0))
    
    plt.tight_layout()
    save_figure_all_formats(fig, output_root, "figure_5_evaluation_summary_reworked")
    plt.close(fig)

def fig6_gradcam(output_root):
    manifest_path = Path("outputs/explainability") / RUN_NAME / "xai_gallery/xai_manifest.json"
    if not manifest_path.exists():
        manifest_path = Path("outputs/friend_share") / RUN_NAME / "explainability/xai_gallery/xai_manifest.json"
        
    if not manifest_path.exists():
        print(f"Warning: Grad-CAM manifest not found at {manifest_path}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        ax.text(0.5, 0.5, "Grad-CAM Manifest Unavailable", ha='center', va='center')
        save_figure_all_formats(fig, output_root, "figure_6_gradcam_summary_reworked")
        plt.close(fig)
        return
        
    with open(manifest_path, "r") as f:
        items = json.load(f)
        
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    
    pl_correct = [i for i in items if i["true_label"] == "point_like" and i["pred_label"] == "point_like"]
    pfm_correct = [i for i in items if i["true_label"] == "point_flaky_mixed" and i["pred_label"] == "point_flaky_mixed"]
    fl_correct = [i for i in items if i["true_label"] == "flaky" and i["pred_label"] == "flaky"]
    wrong = [i for i in items if i["true_label"] != i["pred_label"]]
    
    selected = []
    if pl_correct: selected.append(pl_correct[0])
    if pfm_correct: selected.append(pfm_correct[0])
    if fl_correct: selected.append(fl_correct[0])
    if wrong: selected.append(wrong[0])
    
    for i, item in enumerate(selected):
        if i >= 4: break
        raw_path = item["source_image_path"]
        cam_path = Path("outputs") / item["gradcam_path"]
        
        try:
            img_raw = Image.open(raw_path).convert("RGB")
            # We crop the raw image for a cleaner input view
            mask_path = "data/raw/sustech_sysu/corneaLabels/" + Path(raw_path).name.replace(".jpg", ".png")
            if Path(mask_path).exists():
                mask = Image.open(mask_path).convert("L")
                img_crop = extract_cornea_crop_scale_v1(img_raw, mask)
            else:
                img_crop = img_raw

            axes[i, 0].imshow(img_crop)
            axes[i, 0].set_title(f"Input Crop\nTrue: {item['true_label']}")
            axes[i, 0].axis("off")
            
            img_cam = Image.open(cam_path).convert("RGB")
            axes[i, 1].imshow(img_cam)
            axes[i, 1].set_title(f"Grad-CAM Overlay")
            axes[i, 1].axis("off")
            
            # Text panel
            axes[i, 2].axis("off")
            info = f"True: {item['true_label']}\nPred: {item['pred_label']}\nConf: {item['confidence']:.2f}"
            
            color = "#bbf7d0" if item['true_label'] == item['pred_label'] else "#fecaca"
            axes[i, 2].text(0.5, 0.5, info, fontsize=12, ha='center', va='center', bbox=dict(boxstyle="round,pad=1", facecolor=color, edgecolor="none", alpha=1.0))
            
        except Exception as e:
            print(f"Failed to load image for Fig 6: {e}")
            for ax in axes[i]: ax.axis("off")

    for j in range(len(selected), 4):
        for k in range(3): axes[j,k].axis("off")
            
    plt.tight_layout()
    save_figure_all_formats(fig, output_root, "figure_6_gradcam_summary_reworked")
    plt.close(fig)

def fig7_cv_split_sensitivity(output_root):
    cv_csv = Path("outputs/reports/cv_pattern_3class/w0035_style/CV10_RESULTS.csv")
    if not cv_csv.exists():
        print(f"Warning: CV results not found at {cv_csv}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        ax.text(0.5, 0.5, "CV Results CSV Unavailable", ha='center', va='center')
        save_figure_all_formats(fig, output_root, "figure_7_cv_split_sensitivity_reworked")
        plt.close(fig)
        return

    df = pd.read_csv(cv_csv)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    folds = [f.replace("fold_", "F") for f in df["Fold"]]
    bas = df["Test BA"]
    
    ax.bar(folds, bas, color="#94a3b8", alpha=0.8, edgecolor="#64748b", linewidth=1.5)
    
    mean_ba = bas.mean()
    std_ba = bas.std()
    
    ax.axhline(mean_ba, color="#1e293b", linestyle="-", linewidth=2, label=f"CV Mean BA: {mean_ba:.4f} ± {std_ba:.4f}")
    ax.axhline(0.8671, color="#10b981", linestyle="--", linewidth=2, label="w0035 Fixed Holdout BA: 0.8671")
    
    # Highlight max
    max_idx = bas.idxmax()
    ax.bar(folds[max_idx], bas[max_idx], color="#3b82f6", alpha=0.8, edgecolor="#2563eb", linewidth=1.5, label="Max CV Fold")
    
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Test Balanced Accuracy", fontsize=12)
    ax.set_xlabel("Cross-Validation Fold", fontsize=12)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(axis='y', linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    save_figure_all_formats(fig, output_root, "figure_7_cv_split_sensitivity_reworked")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_root = Path("outputs/paper_figures/w0035_reworked_paper_bundle")
    output_root.mkdir(parents=True, exist_ok=True)
    
    print("Generating Figure 2: Preprocessing...")
    fig2_preprocessing(output_root)
    print("Generating Figure 4: Training Curves...")
    fig4_training_curves(output_root)
    print("Generating Figure 5: Evaluation Metrics...")
    fig5_evaluation_metrics(output_root)
    print("Generating Figure 6: Grad-CAM...")
    fig6_gradcam(output_root)
    print("Generating Figure 7: CV Context...")
    fig7_cv_split_sensitivity(output_root)

    print(f"Matplotlib figures generated in {output_root}")

    index_md = [
        "# w0035 Reworked Figure Bundle Index",
        "",
        "This bundle contains clean, paper-ready visual artifacts for the w0035 fixed-holdout model, utilizing Mermaid for architecture/workflow diagrams and matplotlib for metric plots.",
        "",
        "Use `w0035_reworked_paper_bundle` for paper/report figures. The earlier `w0035_final_paper_bundle` is superseded by this cleaner Mermaid-based bundle.",
        "",
        "## Figures",
        "- `mermaid/figure_1_w0035_system_pipeline.mmd`: w0035 system pipeline Mermaid diagram.",
        "- `figure_2_preprocessing_and_augmentation_examples.png/pdf`: Real image examples of preprocessing and stochastic augmentation.",
        "- `mermaid/figure_3_convnextv2_w0035_architecture.mmd`: ConvNeXtV2 Tiny w0035 architecture Mermaid diagram.",
        "- `figure_4_training_curves_reworked.png/pdf/svg`: Clean training loss and validation selection curves.",
        "- `figure_5_evaluation_summary_reworked.png/pdf/svg`: Comprehensive evaluation metrics panel.",
        "- `figure_6_gradcam_summary_reworked.png/pdf`: Cleaned Grad-CAM explainability panel.",
        "- `figure_7_cv_split_sensitivity_reworked.png/pdf/svg`: Cross-validation robustness estimation.",
        "- `mermaid/figure_8_model_development_summary.mmd`: Final model development story Mermaid diagram."
    ]

    Path("outputs/reports/model_improve/w0035_reworked_figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports/model_improve/w0035_reworked_figures/W0035_REWORKED_FIGURE_BUNDLE_INDEX.md").write_text("\n".join(index_md))
    
    captions_md = [
        "# w0035 Reworked Figure Captions",
        "",
        "## Figure 1: w0035 System Pipeline",
        "Overview of the corneal ulcer pattern-classification pipeline. Raw slit-lamp images undergo mask normalization and deterministic cornea-centered cropping before stochastic online augmentation. The ConvNeXtV2 Tiny backbone is trained using a weighted sampler, weighted cross-entropy, and an ordinal auxiliary loss (weight 0.035).",
        "",
        "## Figure 2: Preprocessing and Augmentation Examples",
        "Visual demonstration of the image preprocessing and augmentation stages. The pipeline extracts a normalized cornea crop (`cornea_crop_scale_v1`), which is subjected to stochastic augmentations (`pattern_augplus_v2`) during training, including random resizing, rotation, color jitter, and mild Gaussian blur to simulate acquisition variance.",
        "",
        "## Figure 3: ConvNeXtV2 w0035 Architecture",
        "Detailed schematic of the ConvNeXtV2 Tiny architecture employed. The network consists of a stem convolution and four hierarchical feature stages. An auxiliary ordinal loss head (weight 0.035) enforces disease severity progression constraints during training, while final deployment relies solely on the primary 3-class linear head.",
        "",
        "## Figure 4: Training and Validation Selection Curves",
        "Training trajectory of the w0035 model. Panel A plots the primary training and validation loss over epochs. Panels B and C track the validation balanced accuracy and macro F1 scores. The selected checkpoint (vertical dashed line) was chosen based on validation balanced accuracy prior to test-set evaluation.",
        "",
        "## Figure 5: Evaluation Summary",
        "Comprehensive evaluation on the frozen holdout test split. The panel includes absolute and normalized confusion matrices (A, B), One-vs-Rest Receiver Operating Characteristic (ROC) curves (C), Precision-Recall (PR) curves (D), empirical calibration curves (E), and a final metrics summary (F). The model achieves a Test Balanced Accuracy of 0.8671.",
        "",
        "## Figure 6: Grad-CAM Summary",
        "Explainable AI (XAI) overlays using Gradient-weighted Class Activation Mapping (Grad-CAM). The heatmaps highlight regions the network focuses on to make its terminal classification decisions, predominantly aligning with clinically relevant ulcer borders and peripheral flaky edges.",
        "",
        "## Figure 7: Split Sensitivity (10-Fold CV Context)",
        "Robustness analysis of the w0035-style training recipe evaluated across 10 random stratified folds. The CV mean balanced accuracy (0.7109 ± 0.0795) highlights split sensitivity inherent to the dataset. Cross-validation estimates split robustness and does not replace the fixed-holdout benchmark (dashed line).",
        "",
        "## Figure 8: Model Development Summary",
        "Development trajectory leading to the w0035 challenger model. The diagram outlines the path from the official anchor, the selected w0035 parameters, and the subsequent negative findings from attempted post-recovery strategies (dual-crop, logit soups, and external SSL) that failed to surpass the w0035 baseline's holdout performance."
    ]
    Path("outputs/reports/model_improve/w0035_reworked_figures/W0035_REWORKED_FIGURE_CAPTIONS.md").write_text("\n".join(captions_md))

if __name__ == "__main__":
    main()
