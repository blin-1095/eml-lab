from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from torch.optim.adam import Adam

from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import os
import time
import matplotlib.pyplot as plt

from tinyyolov2 import TinyYoloV2
from utils.loss import YoloLoss
from utils.early_stopping import EarlyStopping
from utils.metrics import precision_recall_levels, ap
from utils.yolo import filter_boxes, nms
from utils.logger import ExperimentLogger

# make the linter shut up
from torch.optim.optimizer import Optimizer
from typing_extensions import Tuple, Dict, Optional, Any


class BasePipeline(ABC):
    """
    Standardized base pipeline for training TinyYoloV2.
    Handles the boilerplate training loops, early stopping, and evaluation.
    """

    def __init__(
        self, 
        pipeline_name: str, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        test_loader: DataLoader, 
        device: torch.device, 
        learning_rate: float, 
        epochs: int, 
        sd_path: Optional[str] = None,
        patience: int = 20
    ):
        self.pipeline_name = pipeline_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sd_path = sd_path
        self.patience = patience
        
        # Placeholders for architecture components
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.criterion_train: Optional[nn.Module] = None
        self.criterion_eval: Optional[nn.Module] = None
    
    @abstractmethod
    def _setup_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def _preprocess_batch(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def _train_epoch(self) -> float:

        assert self.model is not None, "Model not initialized"
        assert self.criterion_train is not None, "Train criterion not initialized"
        assert self.optimizer is not None, "Optimizer not initialized"

        self.model.train()
        running_loss = 0.0

        loop = tqdm(self.train_loader, leave=True, desc="Training")
        for batch_idx, (images, targets) in enumerate(loop):
            images = images.to(self.device)
            targets = targets.to(self.device)

            images, targets = self._preprocess_batch(images, targets)
            predictions = self.model(images, yolo=False)
            loss, _ = self.criterion_train(predictions, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        return running_loss / len(self.train_loader)

    def _validate_epoch(self, loader, split_name="Validation") -> float:

        assert self.model is not None, "Model not initialized"
        assert self.criterion_eval is not None, "Eval criterion not initialized"

        self.model.eval()
        running_loss = 0.0

        loop = tqdm(loader, leave=True, desc=split_name)
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(loop):
                images = images.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(images, yolo=False)
                loss, _ = self.criterion_eval(predictions, targets)

                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())

        return running_loss / len(loader)


    def run(self) -> dict[str, Any]:
        """Encapsulates the entire training process to ensure a fresh start each time."""
        print(f"\n{'='*50}")
        print(f"STARTING PIPELINE: {self.pipeline_name}")
        print(f"{'='*50}")

        logger = ExperimentLogger(self.pipeline_name)

        # Initialize network components
        self.model = self._setup_model()
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.criterion_train = YoloLoss(anchors=self.model.anchors)
        self.criterion_eval = YoloLoss(anchors=self.model.anchors)
        self.criterion_eval.seen = 999999

        early_stopper = EarlyStopping(patience=self.patience, min_delta=0.01)
        
        best_state_dict: dict = {}

        for epoch in range(self.epochs):
            print(f"\n--- {self.pipeline_name} | Epoch [{epoch+1}/{self.epochs}] ---")
            
            train_loss = self._train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            
            val_loss = self._validate_epoch(self.val_loader, split_name="Validation")
            print(f"Val Loss: {val_loss:.4f}")

            if val_loss < min(logger.val_losses, default=float('inf')):
                best_state_dict = copy.deepcopy(self.model.state_dict())

            logger.log_epoch(train_loss, val_loss) 
            early_stopper(val_loss)
            
            if early_stopper.early_stop:
                print(f"\n[!] Early stopping triggered at epoch {epoch+1}. No improvement for {early_stopper.patience} epochs.")
                break

        print(f"\nLoading best weights of currently best model for final test...")
        self.model.load_state_dict(best_state_dict)

        print(f"\n--- {self.pipeline_name} | Final Testing Phase ---")
        test_loss = self._validate_epoch(self.test_loader, split_name="Testing")
        print(f"Final Test Loss: {test_loss:.4f}")

        print("\nCalculating Average Precision (AP) and Speed Metrics...")
        metrics_dict = self._evaluate_metrics(self.test_loader)
        current_fps = self._benchmark_inference_speed()
        total_params = sum(p.numel() for p in self.model.parameters())
        current_pruning_ratio = getattr(self, 'pruning_ratio', 0.0)

        print(f"Final AP: {metrics_dict['ap']:.4f} | FPS: {current_fps:.1f} | Params: {total_params}")

        final_results = {
            "train_losses": logger.train_losses,
            "val_losses": logger.val_losses,
            "test_loss": test_loss,
            "ap": metrics_dict["ap"],
            "fps": current_fps,
            "inference_time_ms": 1000.0 / current_fps if current_fps > 0 else 0.0,
            "total_parameters": total_params,
            "test_precision_levels": metrics_dict["precision_levels"],
            "test_recall_levels": metrics_dict["recall_levels"],
            "pruning_ratio": current_pruning_ratio
        }

        logger.log_final_metrics(final_results)
        logger.save_report()

        clean_name = self.pipeline_name.split()[0].lower()
        self._export_model(best_state_dict, dest_path=f"models/{clean_name}_best_model.onnx")
        self._export_state_dict(best_state_dict, dest_path=f"state_dicts/{clean_name}_best_sd.pt")

        self._generate_plot(logger.train_losses, logger.val_losses)

        return final_results 

    def _evaluate_metrics(self, loader) -> dict:
        """Runs inference and calculates Precision, Recall, and AP."""
        assert self.model is not None, "Model not initialized"
        self.model.eval()
        
        test_precision = []
        test_recall = []
        
        loop = tqdm(loader, leave=True, desc="Evaluating AP")
        with torch.no_grad():
            for images, targets in loop:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images, yolo=True)
                outputs = filter_boxes(outputs, 0.25)
                outputs = nms(outputs, 0.5)
                
                # Handle batch sizes correctly
                for i in range(images.size(0)):
                    prec, rec = precision_recall_levels(targets[i], outputs[i])
                    test_precision.append(prec)
                    test_recall.append(rec)
                
        final_ap = ap(test_precision, test_recall)
        
        return {
            "ap": float(final_ap),
            "precision_levels": test_precision,
            "recall_levels": test_recall
        }
    
    def _generate_plot(self, train_losses: list, val_losses: list):
        """Generates a standard standalone plot for this specific pipeline run for debugging purposes."""
        os.makedirs("results/debug", exist_ok=True)
        clean_name = self.pipeline_name.lower().replace(" ", "_")
        filepath = f"results/debug/{clean_name}_loss_curve.png"

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, label='Train Loss', marker='o')
        plt.plot(epochs, val_losses, label='Val Loss', marker='s')
        
        plt.title(f'Loss Curve: {self.pipeline_name}', fontsize=14)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        
        plt.savefig(filepath)
        print(f"[*] Saved training curve to '{filepath}'")
        plt.close() # Close it so it doesn't pop up and pause a headless server script!

        
    def _export_model(self, state_dict: dict, dest_path: str):
        """
        Export the trained model in ONNX format.
        Can export all models in this project.
        """

        # clear vram before onnx export
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        sd = copy.deepcopy(state_dict)

        # Export best model in onnx format
        new_channels = [
            sd['conv1.weight'].shape[0],
            sd['conv2.weight'].shape[0],
            sd['conv3.weight'].shape[0],
            sd['conv4.weight'].shape[0],
            sd['conv5.weight'].shape[0],
            sd['conv6.weight'].shape[0],
            sd['conv7.weight'].shape[0],
            sd['conv8.weight'].shape[0]
        ]

        # Dynamically calculate the number of classes of the given network
        conv9_filters = sd['conv9.weight'].shape[0]
        num_classes = int((conv9_filters / 5) - 5)

        best_net = TinyYoloV2(num_classes=num_classes, channels=new_channels)
        best_net.load_state_dict(state_dict)
        best_net.to(self.device)
        best_net.eval()

        # extract a dummy input for the onnx exporter
        export_input, _ = next(iter(self.test_loader))
        export_input = export_input[:1].to(self.device)

        torch.onnx.export(
            best_net,
            export_input,
            dest_path,
            export_params=True,
            opset_version=11,
            input_names=['input_image'],
            output_names=['yolo_output'],

            # dynamic axes so the model can be used with different batch sizes
            dynamic_axes={'input_image': {0: 'batch_size'}, 'yolo_output': {0: 'batch_size'}}
        )

        print(f"Exported model to '{dest_path}'! (Detected Classes: {num_classes})")
    

    def _export_state_dict(self, state_dict: dict, dest_path: str):
        """
        Export state_dict to the state_dict location in the project.
        """

        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)


        torch.save(state_dict, dest_path)
        
        print(f"Successfully exported state_dict to '{dest_path}'!")

    def _benchmark_inference_speed(self, num_batches: int = 50) -> float:
        """Runs a quick inference benchmark on the test set to calculate FPS."""

        assert self.model is not None, "Model not initialized"

        self.model.eval()
        
        # Warm-up phase (GPU requires a few passes to spin up to max clock speed)
        warmup_input, _ = next(iter(self.test_loader))
        warmup_input = warmup_input.to(self.device)
        with torch.no_grad():
            for _ in range(10):
                self.model(warmup_input)
                
        # Benchmark phase
        total_time = 0.0
        total_images = 0
        
        with torch.no_grad():
            for i, (images, _) in enumerate(self.test_loader):
                if i >= num_batches:
                    break
                    
                images = images.to(self.device)
                batch_size = images.shape[0]
                
                # Sync CUDA before and after to get accurate GPU timings!
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                start_time = time.time()
                self.model(images)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                total_time += (time.time() - start_time)
                total_images += batch_size
                
        fps = total_images / total_time
        return fps
