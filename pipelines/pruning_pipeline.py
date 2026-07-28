import torch

from tinyyolov2 import TinyYoloV2
from utils.pruning import l1_structured_pruning, densify_state_dict

from pipelines.base_pipeline import BasePipeline

# make pyright shut up
from torch.utils.data import DataLoader
from typing_extensions import Optional

class PruningPipeline(BasePipeline):
    
    def __init__(
        self, 
        pipeline_name: str, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        test_loader: DataLoader, 
        device: torch.device, 
        learning_rate: float, 
        epochs: int, 
        pruning_ratio: float,
        patience: int = 20
    ):
        
        super().__init__(
            pipeline_name=pipeline_name, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            test_loader=test_loader, 
            device=device, 
            learning_rate=learning_rate, 
            epochs=epochs, 
            patience=patience
        )

        self.pruning_ratio = pruning_ratio

    def _setup_model(self) -> torch.nn.Module:
        """
        Loads a pre-trained model, applies L1 structured pruning, 
        densifies the architecture, and returns the smaller model for fine-tuning.
        """

        if self._state_dict is None:
            raise ValueError("No state dict loaded. Required for pruning!")
        
        print(f"[*] Pruning.")
        original_sd = self._state_dict

        pruned_sd = l1_structured_pruning(original_sd, self.pruning_ratio)
        dense_sd = densify_state_dict(pruned_sd)

        new_channels = [
            dense_sd['conv1.weight'].shape[0],
            dense_sd['conv2.weight'].shape[0],
            dense_sd['conv3.weight'].shape[0],
            dense_sd['conv4.weight'].shape[0],
            dense_sd['conv5.weight'].shape[0],
            dense_sd['conv6.weight'].shape[0],
            dense_sd['conv7.weight'].shape[0],
            dense_sd['conv8.weight'].shape[0]
        ]

        conv9_filters = dense_sd['conv9.weight'].shape[0]
        num_classes = int((conv9_filters / 5) - 5)

        pruned_model = TinyYoloV2(num_classes=num_classes, channels=new_channels)
        pruned_model.load_state_dict(dense_sd)

        return pruned_model.to(self.device)

    def _preprocess_batch(self, images: torch.Tensor, targets: torch.Tensor):
        """No preprocessing required in the baseline model"""
            
        return images, targets
