import os
import json
import glob
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================

RAW_DATA_DIR = "raw_data"
PLOT_DIR = "plots"

def load_metrics_data():
    """Loads all JSON experiment reports from the raw_data directory."""
    if not os.path.exists(RAW_DATA_DIR):
        print(f"[!] Directory '{RAW_DATA_DIR}' not found.")
        return []

    data = []
    for filepath in glob.glob(os.path.join(RAW_DATA_DIR, "*.json")):
        try:
            with open(filepath, 'r') as f:
                experiment = json.load(f)
                
                # Ensure the necessary keys exist before adding to our plotting list
                if "inference_time_ms" in experiment and "ap" in experiment and "pruning_ratio" in experiment:
                    data.append(experiment)
        except Exception as e:
            print(f"[!] Failed to load {filepath}: {e}")
            
    return data

# ==========================================
# 2. PLOTTING FUNCTIONS
# ==========================================

def plot_pruning_pareto(data):
    """
    Scatter plot to easily identify the optimal model.
    With X = Inference Time and Y = AP, the optimal models are in the TOP-LEFT.
    """
    plt.figure(figsize=(8, 6))
    
    # Extract data for axes
    inf_times = [d["inference_time_ms"] for d in data]
    aps = [d["ap"] for d in data]
    ratios = [d["pruning_ratio"] for d in data]
    
    # Create scatter plot mapped to pruning ratios
    scatter = plt.scatter(inf_times, aps, s=150, c=ratios, cmap='viridis', zorder=5, edgecolors='black')
    
    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Pruning Ratio', fontweight='bold')
    
    # Annotate points with their pruning ratio
    for idx in range(len(data)):
        plt.annotate(
            f"{ratios[idx]:.2f}", 
            (inf_times[idx], aps[idx]), 
            textcoords="offset points", 
            xytext=(0, 12), 
            ha='center', 
            fontweight='bold'
        )

    # Labels and Grid
    plt.xlabel("Inference Time (ms) - Lower is Better", fontweight='bold')
    plt.ylabel("Accuracy (AP) - Higher is Better", fontweight='bold')
    plt.title("Latency vs Accuracy (Pareto Frontier)")
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0) 
    
    # Save Plot
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "pruning_pareto_scatter.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"  -> Saved: {save_path}")

# ---------------------------------------------------------
# 💡 ADD NEW PLOTTING FUNCTIONS BELOW THIS LINE
# ---------------------------------------------------------

# def plot_precision_recall_curves(data):
#     """Example of a future plotting function."""
#     plt.figure()
#     # ... your plotting logic here ...
#     plt.savefig(os.path.join(PLOT_DIR, "prc_curves.png"))
#     plt.close()
#     print(f"  -> Saved: prc_curves.png")


# ==========================================
# 3. MAIN EXECUTION PIPELINE
# ==========================================

def generate_all_plots(data):
    """
    Central pipeline to run all plotting functions.
    Register your new plotting functions here.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    print("[*] Generating Pareto Scatter Plot...")
    plot_pruning_pareto(data)
    
    # -----------------------------------------------------
    # 💡 REGISTER NEW PLOTS HERE
    # -----------------------------------------------------
    # print("[*] Generating Precision-Recall Curves...")
    # plot_precision_recall_curves(data)


if __name__ == "__main__":
    print("[*] Loading data for plotting...")
    experiment_data = load_metrics_data()
    
    if experiment_data:
        print(f"[*] Found {len(experiment_data)} experiments. Generating plots...\n")
        generate_all_plots(experiment_data)
        print("\n[*] All plots generated successfully!")
    else:
        print("[!] No valid data found to plot. Check your raw_data/ directory.")