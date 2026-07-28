import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DATA_DIR = "raw_data"   # Adjust to match your raw data directory path if needed
PLOT_DIR = "results/plots" # Adjust to match your output plots directory if needed

def load_data():
    """Finds and categorizes all JSON experiment reports, separating train logs from evaluation data."""
    data = {
        "baseline": None,
        "fused": None, 
        "quantized": None, 
        "augmented": None,
        "augmented_train": None,
        "person_only": None,
        "person_only_train": None,
        "pruned": [],
        "pruned_train": []
    }
    
    files = glob.glob(f"{DATA_DIR}/*.json")
    for f in files:
        with open(f, 'r') as file:
            report = json.load(file)
            
            filename = os.path.basename(f).lower()
            name = report.get("pipeline_name", filename).lower()
            report["pipeline_name"] = report.get("pipeline_name", filename)
            
            # Differentiate between training loop logs and evaluation reports
            is_train = "train" in filename or "train" in name
            
            # Categorize based on architecture/pipeline type
            if "quantized" in name:
                data["quantized"] = report
            elif "fused" in name:
                data["fused"] = report
            elif "baseline" in name:
                data["baseline"] = report
            elif "augmented" in name:
                if is_train:
                    data["augmented_train"] = report
                else:
                    data["augmented"] = report
            elif "person" in name:
                if is_train:
                    data["person_only_train"] = report
                else:
                    data["person_only"] = report
            elif "pruned" in name:
                if is_train:
                    data["pruned_train"].append(report)
                else:
                    data["pruned"].append(report)
                
    # Sort pruned models by their pruning ratio (0.0 to 0.8)
    data["pruned"] = sorted(data["pruned"], key=lambda x: x.get("pruning_ratio", 0.0))
    data["pruned_train"] = sorted(data["pruned_train"], key=lambda x: x.get("pruning_ratio", 0.0))
    
    return data

def get_all_models(data):
    """Helper to compile a flat list of all valid evaluation reports for comparative plots."""
    models = []
    if data["baseline"]: models.append(data["baseline"])
    if data["fused"]: models.append(data["fused"]) 
    if data["quantized"]: models.append(data["quantized"])
    if data["augmented"]: models.append(data["augmented"])
    if data["person_only"]: models.append(data["person_only"])
    models.extend(data["pruned"])
    return models

# ---------------------------------------------------------
# 2. ORIGINAL & PRUNING PLOTTING FUNCTIONS
# ---------------------------------------------------------
def plot_master_pr_curve(data):
    """Overlays averaged Precision-Recall curves for the main pipelines."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    colors = {"baseline": "tab:blue", "augmented": "tab:orange", "person_only": "tab:green"}
    labels = {"baseline": "Baseline (20 Classes)", 
              "augmented": "Augmented (20 Classes)", 
              "person_only": "Person-Only (1 Class)"}
    
    for key in ["baseline", "augmented", "person_only"]:
        if data[key]:
            rec_raw = data[key].get("test_recall_levels", [])
            prec_raw = data[key].get("test_precision_levels", [])
            ap = data[key].get("ap", 0.0)
            
            if rec_raw and prec_raw:
                rec = np.mean(np.array(rec_raw), axis=0)
                prec = np.mean(np.array(prec_raw), axis=0)
                plt.plot(rec, prec, marker='o', label=f"{labels[key]} | AP: {ap:.3f}", color=colors[key], linewidth=2)

    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.xlabel("Recall", fontweight='bold')
    plt.ylabel("Precision", fontweight='bold')
    plt.title("Master Precision-Recall Curve Comparison")
    plt.legend(loc="lower left", frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/master_pr_curve.png", dpi=300)
    plt.close()

def plot_convergence_overlay(data):
    """Compares how fast the main pipelines learned (Validation Loss) using training logs."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    configs = [
        ("Baseline", data.get("baseline"), "tab:blue"),
        ("Augmented", data.get("augmented_train"), "tab:orange"),
        ("Person-Only", data.get("person_only_train"), "tab:green")
    ]
    
    for label, model_data, color in configs:
        if model_data:
            val_losses = model_data.get("val_losses", [])
            if val_losses:
                epochs = range(1, len(val_losses) + 1)
                plt.plot(epochs, val_losses, marker='s', label=model_data.get("pipeline_name", label), color=color, linewidth=2)

    plt.xlabel("Epoch", fontweight='bold')
    plt.ylabel("Validation Loss", fontweight='bold')
    plt.title("Convergence Speed (Validation Loss Overlay)")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/convergence_overlay.png", dpi=300)
    plt.close()

def plot_dual_axis_pruning(pruned_data):
    """Shows AP going down while FPS goes up."""
    if not pruned_data: return
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    ratios = [d["pruning_ratio"] for d in pruned_data]
    aps = [d["ap"] for d in pruned_data]
    fps = [d["fps"] for d in pruned_data]
    
    fig, ax1 = plt.subplots(figsize=(9, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Pruning Ratio', fontweight='bold')
    ax1.set_ylabel('Average Precision (AP)', color=color, fontweight='bold')
    ax1.plot(ratios, aps, color=color, marker='o', linewidth=2, label="AP")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Frames Per Second (FPS)', color=color, fontweight='bold')
    ax2.plot(ratios, fps, color=color, marker='s', linestyle='--', linewidth=2, label="FPS")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(False)

    plt.title("The Pruning Trade-off: Accuracy vs. Speed")
    fig.tight_layout()
    plt.savefig(f"{PLOT_DIR}/pruning_dual_axis.png", dpi=300)
    plt.close()

def plot_pruning_pareto(pruned_data):
    """Scatter plot to easily identify the optimal model (top right)."""
    if not pruned_data: return
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    
    fps = [d["fps"] for d in pruned_data]
    aps = [d["ap"] for d in pruned_data]
    ratios = [d["pruning_ratio"] for d in pruned_data]
    
    scatter = plt.scatter(fps, aps, s=150, c=ratios, cmap='viridis', zorder=5, edgecolors='black')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Pruning Ratio', fontweight='bold')
    
    for idx in range(len(pruned_data)):
        plt.annotate(f"{ratios[idx]:.2f}", (fps[idx], aps[idx]), 
                     textcoords="offset points", xytext=(0,12), ha='center', fontweight='bold')

    plt.xlabel("Speed (FPS)", fontweight='bold')
    plt.ylabel("Accuracy (AP)", fontweight='bold')
    plt.title("Hardware vs Accuracy (Pareto Frontier)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/pruning_pareto_scatter.png", dpi=300)
    plt.close()

def plot_efficiency_scores(pruned_data):
    """Calculates and plots the custom Efficiency Score."""
    if not pruned_data: return
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    baseline_fps = pruned_data[0]["fps"] if pruned_data[0]["fps"] > 0 else 1.0
    
    ratios = []
    scores = []
    
    for d in pruned_data:
        ratio = d["pruning_ratio"]
        current_fps = d["fps"]
        current_ap = d["ap"]
        
        speedup = current_fps / baseline_fps
        score = current_ap * speedup
        
        ratios.append(ratio)
        scores.append(score)
        
    plt.figure(figsize=(8, 6))
    plt.plot(ratios, scores, marker='*', markersize=14, color='tab:purple', linewidth=2, zorder=3)
    
    max_idx = np.argmax(scores)
    plt.plot(ratios[max_idx], scores[max_idx], marker='o', markersize=22, 
             color='gold', fillstyle='none', mew=2.5, zorder=4)
    plt.annotate("Optimal Model", (ratios[max_idx], scores[max_idx]), 
                 xytext=(0,20), textcoords="offset points", ha='center', fontweight='bold')
    
    plt.xlabel("Pruning Ratio", fontweight='bold')
    plt.ylabel("Efficiency Score (AP × Speedup)", fontweight='bold')
    plt.title("Pruning Efficiency Score (Higher is Better)")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/pruning_efficiency_score.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# 3. OPTIMIZATION COMPARISON PLOTTING FUNCTIONS
# ---------------------------------------------------------
def plot_parameters_vs_inference(data):
    """Parameters vs Inference Time (ms) scatter plot across models."""
    models = get_all_models(data)
    if not models: return
    os.makedirs(PLOT_DIR, exist_ok=True)

    params_list = []
    latency_list = []
    labels = []

    for m in models:
        p = m.get("total_parameters", m.get("params", 0))
        fps = m.get("fps", 1.0)
        latency_ms = (1000.0 / fps) if fps > 0 else 0.0
        
        params_list.append(p / 1e6) # Convert to Millions
        latency_list.append(latency_ms)
        labels.append(m.get("pipeline_name", "Model"))

    plt.figure(figsize=(8, 6))
    plt.scatter(params_list, latency_list, s=120, color='tab:red', zorder=5, edgecolors='black')

    for i, txt in enumerate(labels):
        short_label = txt.replace("Pipeline", "").strip()
        plt.annotate(short_label, (params_list[i], latency_list[i]), 
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    plt.xlabel("Total Parameters (Millions)", fontweight='bold')
    plt.ylabel("Inference Latency (ms per batch)", fontweight='bold')
    plt.title("Model Size vs. Inference Latency Trade-off")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/parameters_vs_latency.png", dpi=300)
    plt.close()

def plot_final_benchmark_bars(data):
    """Grouped bar chart comparing AP and Test Loss for main pipelines."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    main_keys = ["baseline", "augmented", "person_only"]
    names = []
    aps = []
    losses = []

    for k in main_keys:
        if data.get(k):
            names.append(k.capitalize())
            aps.append(data[k].get("ap", 0.0))
            losses.append(data[k].get("test_loss", 0.0))

    if not names: return

    x = np.arange(len(names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 6))

    rects1 = ax1.bar(x - width/2, aps, width, label='AP (Higher=Better)', color='tab:blue')
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, losses, width, label='Test Loss (Lower=Better)', color='tab:orange')

    ax1.set_ylabel('Average Precision (AP)', color='tab:blue', fontweight='bold')
    ax2.set_ylabel('Test Loss', color='tab:orange', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontweight='bold')
    ax1.set_ylim(0, 1.1)

    plt.title("Final Benchmark Performance Comparison")
    fig.tight_layout()
    plt.savefig(f"{PLOT_DIR}/final_benchmark_bars.png", dpi=300)
    plt.close()

def plot_training_gap(data):
    """Smoothed learning curves showing Train vs Val loss with an overfitting gap fill using training logs."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    model = data.get("augmented_train") if data.get("augmented_train") else data.get("person_only_train")
    if not model:
        model = data.get("baseline")
    if not model: return

    train_losses = model.get("train_losses", [])
    val_losses = model.get("val_losses", [])

    if not train_losses or not val_losses: return

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='tab:blue', marker='o', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='tab:orange', marker='s', linewidth=2)
    
    plt.fill_between(epochs, train_losses, val_losses, color='gray', alpha=0.25, label='Generalization Gap')

    plt.xlabel("Epoch", fontweight='bold')
    plt.ylabel("Loss", fontweight='bold')
    plt.title(f"Training Dynamics & Generalization Gap ({model.get('pipeline_name', 'Model')})")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/training_overfitting_gap.png", dpi=300)
    plt.close()

def plot_shaded_auc_pr_curve(data):
    """Precision-Recall curve with shaded area under the curve (AUC / AP) for the best model."""
    models = get_all_models(data)
    if not models: return
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    best_model = max(models, key=lambda x: x.get("ap", 0.0))
    
    rec_raw = best_model.get("test_recall_levels", [])
    prec_raw = best_model.get("test_precision_levels", [])
    
    if not rec_raw or not prec_raw: return

    rec = np.mean(np.array(rec_raw), axis=0)
    prec = np.mean(np.array(prec_raw), axis=0)
    ap = best_model.get("ap", 0.0)

    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, marker='o', color='tab:blue', linewidth=2, label=f"AP = {ap:.3f}")
    
    plt.fill_between(rec, prec, alpha=0.3, color='tab:blue', label='Area Under Curve (AP)')

    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.xlabel("Recall", fontweight='bold')
    plt.ylabel("Precision", fontweight='bold')
    plt.title(f"Precision-Recall AUC Visualization ({best_model['pipeline_name']})")
    plt.legend(loc="lower left", frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/shaded_auc_pr_curve.png", dpi=300)
    plt.close()

def plot_fusion_before_after(data):
    """Side-by-side comparison of AP (identical) vs Inference Time (improved) for BatchNorm Fusion."""
    baseline = data.get("baseline")
    fused = data.get("fused")
    
    if not baseline or not fused: return
    os.makedirs(PLOT_DIR, exist_ok=True)
        
    labels = ['Unfused (Baseline)', 'Fused (Optimized)']
    aps = [baseline.get("ap", 0.0), fused.get("ap", 0.0)]
    
    base_fps = baseline.get("fps", 1.0)
    fused_fps = fused.get("fps", 1.0)
    times = [1000.0 / base_fps if base_fps > 0 else 0, 
             1000.0 / fused_fps if fused_fps > 0 else 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.bar(labels, aps, color=['tab:blue', 'tab:green'], width=0.5, edgecolor='black')
    ax1.set_title("Average Precision (AP)\n(Accuracy Unchanged)", fontweight='bold')
    ax1.set_ylabel("AP Score", fontweight='bold')
    ax1.set_ylim(0, max(aps) * 1.25) 
    for i, v in enumerate(aps):
        ax1.text(i, v + (max(aps)*0.02), f"{v:.4f}", ha='center', fontweight='bold')
        
    ax2.bar(labels, times, color=['tab:red', 'tab:orange'], width=0.5, edgecolor='black')
    ax2.set_title("Inference Latency\n(Lower is Better)", fontweight='bold')
    ax2.set_ylabel("Milliseconds (ms) per batch", fontweight='bold')
    ax2.set_ylim(0, max(times) * 1.25)
    for i, v in enumerate(times):
        ax2.text(i, v + (max(times)*0.02), f"{v:.2f} ms", ha='center', fontweight='bold')
        
    plt.suptitle("BatchNorm Fusion: Free Performance Optimization", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/fusion_before_after.png", dpi=300)
    plt.close()

def plot_quantization_comparison(data):
    """Side-by-side comparison of AP vs Inference Latency: Baseline vs. INT8 Quantized Model."""
    baseline = data.get("baseline")
    quantized = data.get("quantized")
    
    if not baseline or not quantized: return
    os.makedirs(PLOT_DIR, exist_ok=True)
        
    labels = ['Baseline (FP32)', 'Quantized (INT8)']
    aps = [baseline.get("ap", 0.0), quantized.get("ap", 0.0)]
    
    base_fps = baseline.get("fps", 1.0)
    quant_fps = quantized.get("fps", 1.0)
    times = [1000.0 / base_fps if base_fps > 0 else 0, 
             1000.0 / quant_fps if quant_fps > 0 else 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.bar(labels, aps, color=['tab:blue', 'tab:purple'], width=0.5, edgecolor='black')
    ax1.set_title("Average Precision (AP)\n(Accuracy Trade-off)", fontweight='bold')
    ax1.set_ylabel("AP Score", fontweight='bold')
    ax1.set_ylim(0, max(aps) * 1.25) 
    for i, v in enumerate(aps):
        ax1.text(i, v + (max(aps)*0.02), f"{v:.4f}", ha='center', fontweight='bold')
        
    ax2.bar(labels, times, color=['tab:red', 'tab:orange'], width=0.5, edgecolor='black')
    ax2.set_title("Inference Latency\n(Lower is Better)", fontweight='bold')
    ax2.set_ylabel("Milliseconds (ms) per batch", fontweight='bold')
    ax2.set_ylim(0, max(times) * 1.25)
    for i, v in enumerate(times):
        ax2.text(i, v + (max(times)*0.02), f"{v:.2f} ms", ha='center', fontweight='bold')
        
    plt.suptitle("Static Quantization: FP32 Baseline vs. INT8 Quantized", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/quantization_comparison.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    print("[*] Loading JSON experiment data...")
    data = load_data()
    
    print("[*] Generating comprehensive visualization suite...")
    plot_master_pr_curve(data)
    plot_convergence_overlay(data)
    plot_dual_axis_pruning(data["pruned"])
    plot_pruning_pareto(data["pruned"])
    plot_efficiency_scores(data["pruned"])
    plot_parameters_vs_inference(data)
    plot_final_benchmark_bars(data)
    plot_training_gap(data)
    plot_shaded_auc_pr_curve(data)
    plot_fusion_before_after(data)
    plot_quantization_comparison(data)
    
    print(f"[+] Success! Generated all plots in the '{PLOT_DIR}/' folder.")