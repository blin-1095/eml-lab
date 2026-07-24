import os
import json
import matplotlib.pyplot as plt

class ExperimentLogger:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.train_losses = []
        self.val_losses = []
        self.metrics = {}

    def log_epoch(self, train_loss: float, val_loss: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def log_final_metrics(self, final_results_dict: dict):
        self.metrics.update(final_results_dict)

    def save_report(self):
        os.makedirs("raw_data", exist_ok=True)
        clean_name = self.experiment_name.lower().replace(" ", "_")
        
        # Combine everything into one dictionary
        report = {
            "pipeline_name": self.experiment_name,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            **self.metrics
        }
        
        filepath = f"raw_data/{clean_name}_metrics.json"
        with open(filepath, "w") as f:
            json.dump(report, f, indent=4)
        print(f"[*] Saved experiment report to '{filepath}'")