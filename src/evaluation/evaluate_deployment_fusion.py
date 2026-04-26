import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

def get_probs(csv_path):
    df = pd.read_csv(csv_path)
    # Ensure sorted by image_id
    df["image_id_int"] = df["image_id"].astype(int)
    df = df.sort_values("image_id_int").reset_index(drop=True)
    probs = df[["prob_point_like", "prob_point_flaky_mixed", "prob_flaky"]].values
    return probs, df["target_index"].values, df["image_id"].values

def main():
    official_test = "outputs/tmp_repro_54facdc/predictions/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/test_predictions.csv"
    w0035_test = "outputs/predictions/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed42/test_predictions.csv"
    dual_test = "outputs/model_improve_dualcrop_2026-04-26/predictions/pattern3__convnextv2_tiny__dual_crop_context_v1__officialinit_high_lr__holdout_v1__seed42/test_predictions.csv"
    
    official_val = "outputs/tmp_repro_54facdc/predictions/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42/val_predictions.csv"
    w0035_val = "outputs/predictions/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed42/val_predictions.csv"
    dual_val = "outputs/model_improve_dualcrop_2026-04-26/predictions/pattern3__convnextv2_tiny__dual_crop_context_v1__officialinit_high_lr__holdout_v1__seed42/val_predictions.csv"
    
    try:
        p_off_val, y_val, _ = get_probs(official_val)
        p_w35_val, _, _ = get_probs(w0035_val)
        p_dual_val, _, _ = get_probs(dual_val)
        
        p_off_test, y_test, _ = get_probs(official_test)
        p_w35_test, _, _ = get_probs(w0035_test)
        p_dual_test, _, _ = get_probs(dual_test)
    except FileNotFoundError as e:
        print(f"Missing file: {e}")
        return
        
    def eval_fusion(weights, name):
        w = np.array(weights)
        w = w / np.sum(w)
        
        val_probs = w[0] * p_off_val + w[1] * p_w35_val + w[2] * p_dual_val
        val_preds = np.argmax(val_probs, axis=1)
        val_ba = balanced_accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average='macro')
        
        test_probs = w[0] * p_off_test + w[1] * p_w35_test + w[2] * p_dual_test
        test_preds = np.argmax(test_probs, axis=1)
        test_ba = balanced_accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average='macro')
        test_acc = accuracy_score(y_test, test_preds)
        
        # Recall per class
        recalls = []
        for c in range(3):
            mask = y_test == c
            recalls.append(accuracy_score(y_test[mask], test_preds[mask]))
            
        return {
            "name": name,
            "weights": [round(x, 2) for x in w],
            "val_ba": val_ba,
            "val_f1": val_f1,
            "test_ba": test_ba,
            "test_f1": test_f1,
            "test_acc": test_acc,
            "recalls": recalls
        }
        
    results = []
    results.append(eval_fusion([1, 0, 0], "Official Only"))
    results.append(eval_fusion([0, 1, 0], "w0035 Only"))
    results.append(eval_fusion([0, 0, 1], "Dual Crop Only"))
    results.append(eval_fusion([1, 1, 1], "Equal Average"))
    results.append(eval_fusion([0, 2, 1], "w0035=2, Dual=1"))
    results.append(eval_fusion([1, 2, 1], "Official=1, w0035=2, Dual=1"))
    
    # Grid search for best val BA
    best_val_ba = 0
    best_weights = [0, 1, 0]
    for w0 in np.linspace(0, 1, 11):
        for w1 in np.linspace(0, 1, 11):
            for w2 in np.linspace(0, 1, 11):
                if w0 + w1 + w2 == 0: continue
                w = [w0, w1, w2]
                val_probs = w[0]*p_off_val + w[1]*p_w35_val + w[2]*p_dual_val
                val_ba = balanced_accuracy_score(y_val, np.argmax(val_probs, axis=1))
                if val_ba > best_val_ba:
                    best_val_ba = val_ba
                    best_weights = w
                    
    results.append(eval_fusion(best_weights, "Val-Selected Weights"))
    
    md_lines = ["# Dual Crop Deployment Fusion", "", "| Strategy | Weights (Off, w35, Dual) | Val BA | Val F1 | Test BA | Test F1 | Test Acc | PL Recall | PFM Recall | Flaky Recall |", "|---|---|---|---|---|---|---|---|---|---|"]
    
    for r in results:
        weights_str = str(r["weights"])
        md_lines.append(f"| {r['name']} | {weights_str} | {r['val_ba']:.4f} | {r['val_f1']:.4f} | {r['test_ba']:.4f} | {r['test_f1']:.4f} | {r['test_acc']:.4f} | {r['recalls'][0]:.4f} | {r['recalls'][1]:.4f} | {r['recalls'][2]:.4f} |")
        
    out_path = Path("outputs/reports/model_improve/dual_crop_context/DUAL_CROP_DEPLOYMENT_FUSION.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md_lines))
    print(f"Generated {out_path}")

if __name__ == "__main__":
    main()
