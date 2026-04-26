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
    
    if "logit_point_like" in df.columns:
        logits = df[["logit_point_like", "logit_point_flaky_mixed", "logit_flaky"]].values
    else:
        # Fallback to probabilities if logits are not saved in older predictions
        logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7))
        
    return probs, logits, df["target_index"].values, df["image_id"].values

def main():
    base_dir = Path("outputs")
    
    runs = {
        "official": base_dir / "tmp_repro_54facdc/predictions/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__holdout_v1__seed42",
        "w0035_seed42": base_dir / "predictions/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed42",
        "w0035_seed1": base_dir / "model_improve_w0035_seeds_2026-04-26/predictions/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed1",
        "w0035_seed7": base_dir / "model_improve_w0035_seeds_2026-04-26/predictions/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed7",
        "w0035_seed21": base_dir / "model_improve_w0035_seeds_2026-04-26/predictions/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed21",
        "w0035_seed123": base_dir / "model_improve_w0035_seeds_2026-04-26/predictions/pattern3__convnextv2_tiny__cornea_crop_scale_v1__augplus_v2__weighted_sampler_tempered__officialinit_ordinalaux_w0035__holdout_v1__seed123"
    }
    
    val_probs = {}
    val_logits = {}
    test_probs = {}
    test_logits = {}
    
    for name, path in runs.items():
        try:
            vp, vl, y_val, _ = get_probs(path / "val_predictions.csv")
            tp, tl, y_test, _ = get_probs(path / "test_predictions.csv")
            val_probs[name] = vp
            val_logits[name] = vl
            test_probs[name] = tp
            test_logits[name] = tl
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            
    def eval_soup(keys, use_logits=True):
        if not keys: return None
        
        stacked_val = np.stack([val_logits[k] if use_logits else val_probs[k] for k in keys], axis=0)
        avg_val = np.mean(stacked_val, axis=0)
        val_preds = np.argmax(avg_val, axis=1)
        val_ba = balanced_accuracy_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds, average='macro')
        
        stacked_test = np.stack([test_logits[k] if use_logits else test_probs[k] for k in keys], axis=0)
        avg_test = np.mean(stacked_test, axis=0)
        test_preds = np.argmax(avg_test, axis=1)
        test_ba = balanced_accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average='macro')
        test_acc = accuracy_score(y_test, test_preds)
        
        recalls = []
        for c in range(3):
            mask = y_test == c
            recalls.append(accuracy_score(y_test[mask], test_preds[mask]))
            
        return {
            "name": f"{'Logit' if use_logits else 'Prob'} Soup: {', '.join(keys)}",
            "val_ba": val_ba,
            "val_f1": val_f1,
            "test_ba": test_ba,
            "test_f1": test_f1,
            "test_acc": test_acc,
            "recalls": recalls
        }
        
    results = []
    
    # Combinations to try
    combos = [
        ["w0035_seed42", "w0035_seed21"],
        ["w0035_seed42", "w0035_seed21", "w0035_seed123"],
        ["w0035_seed42", "w0035_seed21", "w0035_seed123", "w0035_seed1", "w0035_seed7"],
        ["official", "w0035_seed42"],
        ["official", "w0035_seed42", "w0035_seed21"]
    ]
    
    for combo in combos:
        for use_logits in [True, False]:
            res = eval_soup(combo, use_logits=use_logits)
            if res: results.append(res)
            
    # Add a simple val-selected weighted official+w0035_seed42
    best_val_ba = 0
    best_w = 0
    p_off_v = val_logits["official"]
    p_w35_v = val_logits["w0035_seed42"]
    for w in np.linspace(0, 1, 21):
        v_blend = w * p_off_v + (1 - w) * p_w35_v
        v_ba = balanced_accuracy_score(y_val, np.argmax(v_blend, axis=1))
        if v_ba > best_val_ba:
            best_val_ba = v_ba
            best_w = w
            
    p_off_t = test_logits["official"]
    p_w35_t = test_logits["w0035_seed42"]
    t_blend = best_w * p_off_t + (1 - best_w) * p_w35_t
    t_preds = np.argmax(t_blend, axis=1)
    
    recalls = []
    for c in range(3):
        mask = y_test == c
        recalls.append(accuracy_score(y_test[mask], t_preds[mask]))
        
    results.append({
        "name": f"Logit Weighted: Official({best_w:.2f}) + w0035({1-best_w:.2f})",
        "val_ba": best_val_ba,
        "val_f1": f1_score(y_val, np.argmax(best_w * p_off_v + (1 - best_w) * p_w35_v, axis=1), average='macro'),
        "test_ba": balanced_accuracy_score(y_test, t_preds),
        "test_f1": f1_score(y_test, t_preds, average='macro'),
        "test_acc": accuracy_score(y_test, t_preds),
        "recalls": recalls
    })
    
    md_lines = ["# Logit Soup Results", "", "| Soup Strategy | Val BA | Val F1 | Test BA | Test F1 | Test Acc | PL Recall | PFM Recall | Flaky Recall |", "|---|---|---|---|---|---|---|---|---|"]
    
    # Sort by val_ba
    results.sort(key=lambda x: x["val_ba"], reverse=True)
    
    for r in results:
        md_lines.append(f"| {r['name']} | {r['val_ba']:.4f} | {r['val_f1']:.4f} | {r['test_ba']:.4f} | {r['test_f1']:.4f} | {r['test_acc']:.4f} | {r['recalls'][0]:.4f} | {r['recalls'][1]:.4f} | {r['recalls'][2]:.4f} |")
        
    out_md = Path("outputs/reports/model_improve/logit_soup_w0035/LOGIT_SOUP_RESULTS.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Generated report at {out_md}")

if __name__ == "__main__":
    main()
