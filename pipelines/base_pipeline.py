# make the linter shut up
from __future__ import annotations
from torch.optim.optimizer import Optimizer
from typing_extensions import Tuple, Dict, Optional, Any


from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from torch.optim.adam import Adam

from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import os
import matplotlib.pyplot as plt

from tinyyolov2 import TinyYoloV2
from utils.loss import YoloLoss
from utils.early_stopping import EarlyStopping
from utils.logger import ExperimentLogger


class BasePipeline(ABC):
    """
    Standardized base pipeline for training TinyYoloV2.
    Handles the boilerplate training loops, early stopping and state dict export.
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
        patience: int = 20
    ):
        self.pipeline_name = pipeline_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience

        self._state_dict = None 
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


    def run(self) -> dict:
        """Encapsulates the entire training process to ensure a fresh start each time."""
        print(f"\n{'='*50}")
        print(f"STARTING PIPELINE: {self.pipeline_name}")
        print(f"{'='*50}")

        logger = ExperimentLogger(self.pipeline_name + "_train")

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
            val_loss = self._validate_epoch(self.val_loader, split_name="Validation")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")                   

            if val_loss < min(logger.val_losses, default=float('inf')):
                best_state_dict = copy.deepcopy(self.model.state_dict())

            logger.log_epoch(train_loss, val_loss) 
            early_stopper(val_loss)
            
            if early_stopper.early_stop:
                print(f"\n[!] Early stopping triggered at epoch {epoch+1}. No improvement for {early_stopper.patience} epochs.")
                break

        logger.save_report()

        self.state_dict = copy.deepcopy(best_state_dict)

        print(f"Training Pipeline Finished!")

        return best_state_dict


    def export_state_dict(self, dest_path: str):
        """
        Export state_dict to the state_dict location in the project.
        """

        if self.state_dict is None:
            raise ValueError("No state dict to export. Have you run training already?")

        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)


        torch.save(self.state_dict, dest_path)
        
        print(f"Successfully exported state_dict to '{dest_path}'!")
    
    def load_state_dict(self, state_dict: Optional[str | Dict[str, Any]] = None):
            """
            Loads a saved state dict into the class.
            Accepts either an in-memory dictionary or a file path (string).
            """
            source = state_dict
            
            if source is None:
                raise ValueError("No state dict provided. Pass a dict or a file path")
                
            if isinstance(source, str):
                if not os.path.exists(source):
                    raise FileNotFoundError(f"State dict file not found at: {source}")
                print(f"Loading state dict from file '{source}' to device '{self.device}'...")
                # weights_only=True prevents arbitrary code execution and silences PyTorch FutureWarnings
                loaded_sd = torch.load(source, map_location=self.device, weights_only=True)
            elif isinstance(source, dict):
                print("Loading state dict directly from memory...")
                loaded_sd = source
            else:
                raise TypeError(f"Expected a file path (str) or state dict (dict), got {type(source).__name__}")
            
            self._state_dict = loaded_sd
            print("State dict loaded successfully!")